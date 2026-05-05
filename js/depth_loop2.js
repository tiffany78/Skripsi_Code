var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// =====================
// PARAMETER UMUM
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var batnas = ee.Image("projects/ee-tiffanytasyaagatha/assets/Batnas");

var numScale = 20;   
var rfParams = {
  numberOfTrees: 80,
  variablesPerSplit: null,
  minLeafPopulation: 2,
  bagFraction: 0.6,
  seed: 42
};

// mask area laut dangkal: 0–25 m
var shallowMask = batnas.lt(0).and(batnas.gte(-25));

// daftar fitur yang benar-benar dipakai model
var featureBands = [
  'B2', 'B3', 'B4', 'B8',
  'B2_B3_ratio', 'B2_B4_ratio', 'B3_B4_ratio',
  'log_B2', 'log_B3', 'log_B4'
];

// periode 3 bulanan
var periods = [
  {label: 'Q1_2025', start: '2025-01-01', end: '2025-04-01'},
  {label: 'Q2_2025', start: '2025-04-01', end: '2025-07-01'},
  {label: 'Q3_2025', start: '2025-07-01', end: '2025-10-01'},
  {label: 'Q4_2025', start: '2025-10-01', end: '2026-01-01'}
];

// =====================
// FUNGSI DASAR
// =====================
function buildComposite(startDate, endDate) {
  var filtered = s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    });

  var composite = filtered.median().clip(buffer);
  return composite.set('n_images', filtered.size());
}

function buildPredictors(startDate, endDate) {
  var composite = buildComposite(startDate, endDate);

  // scaling dulu agar konsisten
  var scaled = composite.select(['B2', 'B3', 'B4', 'B8']).multiply(0.0001);

  var ndwi = scaled.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var waterMask = ndwi.gt(0);

  var eps = 0.0001;
  var blue  = scaled.select('B2').add(eps)
    .updateMask(waterMask)
    .updateMask(shallowMask);

  var green = scaled.select('B3').add(eps)
    .updateMask(waterMask)
    .updateMask(shallowMask);

  var red   = scaled.select('B4').add(eps)
    .updateMask(waterMask)
    .updateMask(shallowMask);

  var nir   = scaled.select('B8').add(eps)
    .updateMask(waterMask)
    .updateMask(shallowMask);

  var predictors = ee.Image.cat([
    blue.rename('B2'),
    green.rename('B3'),
    red.rename('B4'),
    nir.rename('B8'),
    blue.divide(green).rename('B2_B3_ratio'),
    blue.divide(red).rename('B2_B4_ratio'),
    green.divide(red).rename('B3_B4_ratio'),
    blue.log().rename('log_B2'),
    green.log().rename('log_B3'),
    red.log().rename('log_B4')
    // kalau mau, bisa tambah ndwi juga:
    // ndwi.updateMask(waterMask).updateMask(shallowMask).rename('NDWI')
  ]).toFloat();

  return {
    predictors: predictors,
    waterMask: waterMask,
    composite: composite
  };
}

function buildDepthLabel(waterMask) {
  return batnas
    .multiply(-1)
    .rename('depth')
    .toFloat()
    .updateMask(shallowMask)
    .updateMask(waterMask);
}

// bin 0–25 m 
function buildDepthBin(depth) {
  var depthBin = depth.expression(
    "(d > 0 && d <= 5)  ? 1" +
    ": (d <= 10) ? 2" +
    ": (d <= 15) ? 3" +
    ": (d <= 20) ? 4" +
    ": (d <= 25) ? 5" +
    ": 0", {
      d: depth
    }
  ).rename('depth_bin').toInt();

  return depthBin.updateMask(depth.mask());
}

// =====================
// AMBIL TRAINING SAMPLE
// =====================
function collectTrainingSample(startDate, endDate, label, ptsPerBin, keepGeom) {
  var prep = buildPredictors(startDate, endDate);
  var predictors = prep.predictors;
  var waterMask = prep.waterMask;
  var depth = buildDepthLabel(waterMask);
  var depthBin = buildDepthBin(depth);

  var stack = predictors.addBands(depth).addBands(depthBin);

  var sample = stack.stratifiedSample({
    numPoints: 0,
    classBand: 'depth_bin',
    classValues: [1, 2, 3, 4, 5],
    classPoints: [ptsPerBin, ptsPerBin, ptsPerBin, ptsPerBin, ptsPerBin],
    region: buffer,
    scale: numScale,
    seed: 42,
    geometries: keepGeom,
    dropNulls: true,
    tileScale: 4
  }).map(function(f) {
    return f.set('period', label);
  });

  if (keepGeom) {
    sample = sample.map(function(f) {
      var coords = f.geometry().coordinates();
      return f.set({
        lon: coords.get(0),
        lat: coords.get(1)
      });
    });
  }

  return sample;
}

// =====================
// HELPER EVALUASI
// =====================
function groupListToFC(groupList) {
  groupList = ee.List(groupList);
  return ee.FeatureCollection(groupList.map(function(d) {
    d = ee.Dictionary(d);
    var bin = ee.Number(d.get('bin')).toInt();

    var minD = bin.subtract(1).multiply(5);
    var maxD = bin.multiply(5);

    return ee.Feature(null, d.combine({
      bin: bin,
      range: minD.format().cat('-').cat(maxD.format())
    }, true));
  }));
}

// =====================
// TRAIN + EVALUASI 1 KUARTAL
// =====================
function runQuarterModel(startDate, endDate, label, ptsPerBin, showLayer) {
  var prep = buildPredictors(startDate, endDate);
  var predictors = prep.predictors;
  var nImages = prep.composite.get('n_images');

  var allSamples = collectTrainingSample(startDate, endDate, label, ptsPerBin, false);

  var split = 0.7;
  var sampleRand = allSamples.randomColumn('rand', 42);
  var trainSet = sampleRand.filter(ee.Filter.lt('rand', split));
  var testSet = sampleRand.filter(ee.Filter.gte('rand', split));

  var rf = ee.Classifier.smileRandomForest(rfParams).setOutputMode('REGRESSION');

  var trainedRF = rf.train({
    features: trainSet,
    classProperty: 'depth',
    inputProperties: featureBands
  });

  var testPred = testSet.classify(trainedRF);

  var corr = testPred.reduceColumns({
    reducer: ee.Reducer.pearsonsCorrelation(),
    selectors: ['depth', 'classification']
  });

  var r = ee.Number(corr.get('correlation'));
  var r2 = r.pow(2);

  var testMetrics = testPred.map(function(f) {
    var a = ee.Number(f.get('depth'));
    var p = ee.Number(f.get('classification'));
    return f.set({
      absError: a.subtract(p).abs(),
      sqError: a.subtract(p).pow(2)
    });
  });

  var mae = ee.Number(testMetrics.aggregate_mean('absError'));
  var rmse = ee.Number(testMetrics.aggregate_mean('sqError')).sqrt();

  print('==============================');
  print('Periode:', label);
  print('Jumlah citra bersih:', nImages);
  print('N total:', allSamples.size());
  print('N train:', trainSet.size());
  print('N test :', testSet.size());
  print('R²:', r2);
  print('RMSE:', rmse);
  print('MAE:', mae);

  // Scatter chart
  var chart = ui.Chart.feature.byFeature(
    testPred,
    'depth',
    ['classification']
  )
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Actual vs Predicted Depth - ' + label,
    hAxis: {title: 'Actual Depth'},
    vAxis: {title: 'Predicted Depth'},
    pointSize: 3,
    trendlines: {0: {showR2: true, visibleInLegend: true}}
  });
  print(chart);

  // RMSE per bin
  var testBinned = testPred.map(function(f) {
    var a = ee.Number(f.get('depth'));
    var p = ee.Number(f.get('classification'));

    var bin = ee.Number(
      ee.Algorithms.If(a.lte(5), 1,
      ee.Algorithms.If(a.lte(10), 2,
      ee.Algorithms.If(a.lte(15), 3,
      ee.Algorithms.If(a.lte(20), 4, 5))))
    );

    var sqError = a.subtract(p).pow(2);

    return f.set({
      bin: bin,
      sqError: sqError,
      one: 1
    });
  });

  var countGroup = testBinned.reduceColumns({
    reducer: ee.Reducer.sum().group({
      groupField: 0,
      groupName: 'bin'
    }),
    selectors: ['bin', 'one']
  });

  var countFC = groupListToFC(countGroup.get('groups'))
    .select(['bin', 'range', 'sum'], ['bin', 'range', 'n']);

  var mseGroup = testBinned.reduceColumns({
    reducer: ee.Reducer.mean().group({
      groupField: 0,
      groupName: 'bin'
    }),
    selectors: ['bin', 'sqError']
  });

  var rmseFC = groupListToFC(mseGroup.get('groups'))
    .map(function(f) {
      return f.set('rmse', ee.Number(f.get('mean')).sqrt());
    })
    .select(['bin', 'rmse']);

  var joined = ee.Join.inner();
  var joinBin = ee.Filter.equals({
    leftField: 'bin',
    rightField: 'bin'
  });

  var rmseTable = ee.FeatureCollection(joined.apply(countFC, rmseFC, joinBin))
    .map(function(f) {
      f = ee.Feature(f);
      return ee.Feature(f.get('primary'))
        .copyProperties(ee.Feature(f.get('secondary')));
    })
    .sort('bin');

  print('RMSE per depth bin - ' + label, rmseTable);

  var rmseChart = ui.Chart.feature.byFeature(rmseTable, 'range', ['rmse'])
    .setChartType('ColumnChart')
    .setOptions({
      title: 'RMSE per Depth Bin - ' + label,
      hAxis: {title: 'Depth bin (m)'},
      vAxis: {title: 'RMSE (m)'},
      legend: {position: 'none'}
    });
  print(rmseChart);

  // Residual plot
  var residuals = testPred.map(function(f){
    var actual = ee.Number(f.get('depth'));
    var pred = ee.Number(f.get('classification'));
    return f.set('residual', actual.subtract(pred));
  });

  var residualChart = ui.Chart.feature.byFeature(
    residuals,
    'depth',
    ['residual']
  )
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Residual Plot - ' + label,
    hAxis: {title: 'Actual Depth'},
    vAxis: {title: 'Residual (Actual - Predicted)'},
    pointSize: 3
  });
  print(residualChart);

  // Prediksi raster kuartal tsb
  var depth_est = predictors
    .classify(trainedRF)
    .rename('Depth_RF')
    .updateMask(shallowMask)
    .max(0)
    .min(25);

  Map.addLayer(
    depth_est,
    {min: 0, max: 25, palette: ['#E0F3FC','#9BC8DB','#569DB9','#084C67']},
    'Prediksi ' + label,
    showLayer
  );

  return {
    model: trainedRF,
    prediction: depth_est,
    testPred: testPred
  };
}

// =====================
// JALANKAN SEMUA KUARTAL
// =====================
var ptsPerBin = 400;

periods.forEach(function(p, idx) {
  runQuarterModel(p.start, p.end, p.label, ptsPerBin, idx === 0);
});