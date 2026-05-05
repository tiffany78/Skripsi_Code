var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// =====================
// PARAMETER UMUM
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var gebco2025 = ee.Image("projects/ee-tiffanytasyaagatha/assets/gebco2025");

var numScale = 20;

// shallow water 0–25 m
var shallowMask = gebco2025.lt(0).and(gebco2025.gte(-25));

// proxy deep water untuk Lyzenga
// kalau bisa, nanti ganti dengan polygon deep-water manual yang lebih aman
var deepWaterMask = gebco2025.lte(-40);

// =====================
// FUNGSI DASAR
// =====================
function buildComposite(startDate, endDate) {
  return s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    })
    .median()
    .clip(buffer);
}

// reflectance 0–1 + water mask
function buildWaterReflectance(startDate, endDate) {
  var composite = buildComposite(startDate, endDate);

  var scaled = composite.select(['B2', 'B3', 'B4', 'B8']).multiply(0.0001);

  var ndwi = scaled.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var waterMask = ndwi.gt(0).and(shallowMask);

  var img = scaled.updateMask(waterMask).updateMask(shallowMask).toFloat();

  return {
    composite: composite,
    scaled: scaled,
    img: img,
    waterMask: waterMask
  };
}

function buildDepthLabel(waterMask) {
  return gebco2025
    .multiply(-1)
    .rename('depth')
    .toFloat()
    .updateMask(shallowMask)
    .updateMask(waterMask);
}

// bin 0–35 m jadi 7 kelas
function buildDepthBin(depth) {
  var depthBin = depth.expression(
    "(d > 0 && d <= 5)  ? 1" +
    ": (d <= 10) ? 2" +
    ": (d <= 15) ? 3" +
    ": (d <= 20) ? 4" +
    ": (d <= 25) ? 5" +
    ": 0", {d: depth}
  ).rename('depth_bin').toInt();

  return depthBin.updateMask(depth.mask());
}

// evaluasi umum
function evaluateRegression(testPred, actualProp, predProp, label) {
  var corr = testPred.reduceColumns({
    reducer: ee.Reducer.pearsonsCorrelation(),
    selectors: [actualProp, predProp]
  });

  var r = ee.Number(corr.get('correlation'));
  var r2 = r.pow(2);

  var metrics = testPred.map(function(f) {
    var a = ee.Number(f.get(actualProp));
    var p = ee.Number(f.get(predProp));
    return f.set({
      absError: a.subtract(p).abs(),
      sqError: a.subtract(p).pow(2)
    });
  });

  var mae = ee.Number(metrics.aggregate_mean('absError'));
  var rmse = ee.Number(metrics.aggregate_mean('sqError')).sqrt();

  print('======================');
  print(label);
  print('R²  :', r2);
  print('RMSE:', rmse);
  print('MAE :', mae);

  var chart = ui.Chart.feature.byFeature(
    testPred, actualProp, [predProp]
  )
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Actual vs Predicted - ' + label,
    hAxis: {title: 'Actual depth'},
    vAxis: {title: 'Predicted depth'},
    pointSize: 3,
    trendlines: {0: {showR2: true, visibleInLegend: true}}
  });
  print(chart);
}

// =====================
// LYZENGA LOG-LINEAR
// =====================
function runLyzenga(startDate, endDate, label, ptsPerBin, showLayer) {
  var prep = buildWaterReflectance(startDate, endDate);
  var img = prep.img;
  var waterMask = prep.waterMask;

  var depth = buildDepthLabel(waterMask);
  var depthBin = buildDepthBin(depth);

  // estimasi deep-water mean
  // lebih bagus kalau diganti ROI manual deep water
  var deepStats = img.updateMask(deepWaterMask).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: buffer,
    scale: 20,
    maxPixels: 1e13,
    tileScale: 4
  });

  var b2inf = ee.Number(deepStats.get('B2'));
  var b3inf = ee.Number(deepStats.get('B3'));
  var b4inf = ee.Number(deepStats.get('B4'));

  print('Deep-water means - ' + label, {
    B2_inf: b2inf,
    B3_inf: b3inf,
    B4_inf: b4inf
  });

  var eps = 1e-6;
  var Xb = img.select('B2').subtract(b2inf).max(eps).log().rename('Xb');
  var Xg = img.select('B3').subtract(b3inf).max(eps).log().rename('Xg');
  var Xr = img.select('B4').subtract(b4inf).max(eps).log().rename('Xr');

  // mulai dari blue + green dulu
  var stack = ee.Image.constant(1).rename('constant')
    .addBands(Xb)
    .addBands(Xg)
    .addBands(Xr) 
    .addBands(depth)
    .addBands(depthBin);

  var sample = stack.stratifiedSample({
    numPoints: 0,
    classBand: 'depth_bin',
    classValues: [1,2,3,4,5],
    classPoints: [ptsPerBin,ptsPerBin,ptsPerBin,ptsPerBin,ptsPerBin],
    region: buffer,
    scale: numScale,
    seed: 42,
    geometries: false,
    dropNulls: true,
    tileScale: 4
  });

  var split = 0.7;
  var rand = sample.randomColumn('rand', 42);
  var trainSet = rand.filter(ee.Filter.lt('rand', split));
  var testSet  = rand.filter(ee.Filter.gte('rand', split));

  // linear regression: depth = a0 + a1*Xb + a2*Xg
  var lr = trainSet.reduceColumns({
    reducer: ee.Reducer.linearRegression({
      numX: 4, 
      numY: 1
    }),
    selectors: ['constant', 'Xb', 'Xg', 'Xr','depth']
  });

  var coeff = ee.Array(lr.get('coefficients'));
  var a0 = ee.Number(coeff.get([0, 0]));
  var a1 = ee.Number(coeff.get([1, 0]));
  var a2 = ee.Number(coeff.get([2, 0]));
  var a3 = ee.Number(coeff.get([3, 0]));

  print('Lyzenga coefficients - ' + label, {
    intercept: a0,
    coef_B2: a1,
    coef_B3: a2,
    coef_B4: a3
  });

  var testPred = testSet.map(function(f) {
    var xb = ee.Number(f.get('Xb'));
    var xg = ee.Number(f.get('Xg'));
    var xr = ee.Number(f.get('Xr'));
    var pred = a0.add(a1.multiply(xb)).add(a2.multiply(xg)).add(a3.multiply(xr));
    return f.set('pred_lyzenga', pred);
  });

  evaluateRegression(testPred, 'depth', 'pred_lyzenga', 'Lyzenga - ' + label);

  var depthLyzenga = ee.Image.constant(a0)
    .add(Xb.multiply(a1))
    .add(Xg.multiply(a2))
    .add(Xr.multiply(a3))
    .rename('Depth_Lyzenga')
    .updateMask(shallowMask)
    .max(0).min(25);

  Map.addLayer(
    depthLyzenga,
    {min: 0, max: 25, palette: ['#E0F3FC','#9BC8DB','#569DB9','#084C67']},
    'Lyzenga ' + label,
    showLayer
  );

  return {
    prediction: depthLyzenga,
    testPred: testPred,
    intercept: a0,
    coefB2: a1,
    coefB3: a2, 
    coefB4: a3
  };
}

var resultLyzenga = runLyzenga(
  '2025-01-01', '2026-01-01',
  'Median_2025',
  400,
  true
);