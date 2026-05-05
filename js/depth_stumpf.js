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
// STUMPF
// =====================
function runStumpf(startDate, endDate, label, ptsPerBin, showLayer) {
  var prep = buildWaterReflectance(startDate, endDate);
  var img = prep.img;
  var waterMask = prep.waterMask;

  var depth = buildDepthLabel(waterMask);
  var depthBin = buildDepthBin(depth);

  var n = 1000;
  var eps = 1e-6;

  var stumpfX = img.expression(
    'log(n * b) / log(n * g)', {
      n: n,
      b: img.select('B2').add(eps),
      g: img.select('B3').add(eps)
    }
  ).rename('X');

  var stack = ee.Image.constant(1).rename('constant')
    .addBands(stumpfX)
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

  // linear regression: depth = a + b*X
  var lr = trainSet.reduceColumns({
    reducer: ee.Reducer.linearRegression({
      numX: 2, // constant + X
      numY: 1
    }),
    selectors: ['constant', 'X', 'depth']
  });

  var coeff = ee.Array(lr.get('coefficients'));
  var a = ee.Number(coeff.get([0, 0]));
  var b = ee.Number(coeff.get([1, 0]));

  print('Stumpf coefficients - ' + label, {
    intercept: a,
    slope: b
  });

  var testPred = testSet.map(function(f) {
    var x = ee.Number(f.get('X'));
    var pred = a.add(b.multiply(x));
    return f.set('pred_stumpf', pred);
  });

  evaluateRegression(testPred, 'depth', 'pred_stumpf', 'Stumpf - ' + label);

  var depthStumpf = ee.Image.constant(a)
    .add(stumpfX.multiply(b))
    .rename('Depth_Stumpf')
    .updateMask(shallowMask)
    .max(0).min(25);

  Map.addLayer(
    depthStumpf,
    {min: 0, max: 25, palette: ['#E0F3FC','#9BC8DB','#569DB9','#084C67']},
    'Stumpf ' + label,
    showLayer
  );

  return {
    prediction: depthStumpf,
    testPred: testPred,
    intercept: a,
    slope: b
  };
}

var resultStumpf = runStumpf(
  '2025-01-01', '2026-01-01',
  'Median_2025',
  400,
  true
);