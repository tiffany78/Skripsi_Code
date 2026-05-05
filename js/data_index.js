// =====================
// 1. MASK HIGH SEDIMENT
// mengikuti definisi visual Anda:
// high sediment = muncul > 1 quarter
// =====================
var highSedMask = highSedFreq.gt(1);
var lowSedMask  = highSedFreq.unmask(0).lte(1);

Map.addLayer(
  highSedMask.selfMask(),
  {palette: ['#d94701']},
  'Excluded High Sediment',
  false
);

Map.addLayer(
  lowSedMask.selfMask(),
  {palette: ['#1a9850']},
  'Low Sediment Area',
  false
);

// =====================
// 2. COMPOSITE UNTUK PEMETAAN KEDALAMAN
// pakai median tahunan, bisa diganti periodenya
// =====================
var mapComposite = s2
  .filterDate('2025-01-01', '2026-01-01')
  .filterBounds(buffer)
  .linkCollection(csPlus, [QA_BAND])
  .map(function(img) {
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
  })
  .median()
  .clip(buffer);

Map.addLayer(
  mapComposite,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500},
  'RGB Mapping Composite',
  false
);

// =====================
// 3. WATER MASK DARI COMPOSITE PEMETAAN
// =====================
var ndwiMap = mapComposite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var waterMaskMap = ndwiMap.gt(0);

var ndwiWaterOnly = ndwiMap.updateMask(waterMaskMap);

Map.addLayer(
  ndwiWaterOnly,
  {
    min: 0,
    max: 0.6,
    palette: ['#f7fcfd', '#ccece6', '#66c2a4', '#238b45', '#00441b']
  },
  'NDWI Water Only - Mapping Composite',
  false
);

// =====================
// BATNAS SEBAGAI TARGET DEPTH
// ubah negatif -> positif
// contoh: -12 m menjadi 12
// =====================
var batnasRaw = ee.Image("projects/ee-tiffanytasyaagatha/assets/Batnas");

// kedalaman positif
var batnasDepth = batnasRaw.multiply(-1).rename('depth');

// batasi range kedalaman yang mau dipetakan
// contoh di sini 0 - 35 m
var minDepth = 0;
var maxDepth = 35;

var depthMask = batnasRaw.lte(-minDepth).and(batnasRaw.gte(-maxDepth));

// =====================
// 5. MASK FINAL UNTUK ANALISIS
// hanya air + low sediment + depth valid
// =====================
var validMask = waterMaskMap
  .and(lowSedMask)
  .and(depthMask);

Map.addLayer(
  validMask.selfMask(),
  {palette: ['#00ff00']},
  'Valid Area for Bathymetry RF',
  false
);

// BATNAS hanya di area valid
var batnasMasked = batnasDepth.updateMask(validMask);

Map.addLayer(
  batnasMasked,
  {
    min: minDepth,
    max: maxDepth,
    palette: ['#08306b', '#2171b5', '#41b6c4', '#a1dab4', '#ffffcc', '#fdae61', '#d73027']
  },
  'BATNAS Depth (Low Sediment Only)',
  true
);

// =====================
// 6. PREDICTOR UNTUK RANDOM FOREST
// gunakan band tampak + rasio sederhana
// =====================
var ratio_B2_B3 = mapComposite.select('B2')
  .divide(mapComposite.select('B3').add(1))
  .rename('ratio_B2_B3');

var ratio_B3_B4 = mapComposite.select('B3')
  .divide(mapComposite.select('B4').add(1))
  .rename('ratio_B3_B4');

var predictors = mapComposite
  .select(['B1', 'B2', 'B3', 'B4'])
  .addBands(ratio_B2_B3)
  .addBands(ratio_B3_B4)
  .updateMask(validMask);

var predictorNames = ['B1', 'B2', 'B3', 'B4', 'ratio_B2_B3', 'ratio_B3_B4'];

// =====================
// 7. STACK TRAINING
// target = depth BATNAS
// =====================
var trainingStack = predictors.addBands(batnasDepth.updateMask(validMask));

// sample dibuat ringan supaya tidak kena memory limit
var samples = trainingStack.sample({
  region: buffer.geometry(),
  scale: 20,
  numPixels: 4000,
  seed: 42,
  geometries: false,
  dropNulls: true
});

print('Jumlah sample', samples.size());
print('Contoh sample', samples.limit(5));

// =====================
// 8. SPLIT TRAIN / TEST
// =====================
var samplesRnd = samples.randomColumn('rand', 42);

var train = samplesRnd.filter(ee.Filter.lt('rand', 0.7));
var test  = samplesRnd.filter(ee.Filter.gte('rand', 0.7));

print('Train size', train.size());
print('Test size', test.size());

// =====================
// 9. RANDOM FOREST REGRESSION
// =====================
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.7,
  seed: 42
})
.setOutputMode('REGRESSION')
.train({
  features: train,
  classProperty: 'depth',
  inputProperties: predictorNames
});

// =====================
// 10. PREDIKSI KEDALAMAN
// hanya pada area valid (non-high-sediment)
// =====================
var depthPred = predictors
  .classify(rf, 'depth_pred')
  .rename('depth_pred')
  .max(minDepth)
  .min(maxDepth)
  .updateMask(validMask);

Map.addLayer(
  depthPred,
  {
    min: minDepth,
    max: maxDepth,
    palette: ['#08306b', '#2171b5', '#41b6c4', '#a1dab4', '#ffffcc', '#fdae61', '#d73027']
  },
  'RF Predicted Depth (Low Sediment Only)',
  true
);

// =====================
// 11. EVALUASI SEDERHANA
// =====================
var testPred = test.classify(rf, 'depth_pred').map(function(f) {
  var actual = ee.Number(f.get('depth'));
  var pred   = ee.Number(f.get('depth_pred'));
  var err    = pred.subtract(actual);

  return f.set({
    sqerr: err.pow(2),
    abserr: err.abs()
  });
});

var rmse = ee.Number(testPred.aggregate_mean('sqerr')).sqrt();
var mae  = ee.Number(testPred.aggregate_mean('abserr'));

var meanActual = ee.Number(testPred.aggregate_mean('depth'));

var testPredR2 = testPred.map(function(f) {
  var actual = ee.Number(f.get('depth'));
  return f.set('sst', actual.subtract(meanActual).pow(2));
});

var sse = ee.Number(testPredR2.aggregate_sum('sqerr'));
var sst = ee.Number(testPredR2.aggregate_sum('sst'));
var r2  = ee.Number(1).subtract(sse.divide(sst));

print('RMSE', rmse);
print('MAE', mae);
print('R2', r2);

// =====================
// 12. CHART OPSIONAL
// =====================
print(
  ui.Chart.feature.byFeature(testPredR2.limit(1000), 'depth', ['depth_pred'])
    .setChartType('ScatterChart')
    .setOptions({
      title: 'Observed vs Predicted Depth',
      hAxis: {title: 'Observed Depth (BATNAS)'},
      vAxis: {title: 'Predicted Depth'},
      pointSize: 3
    })
);