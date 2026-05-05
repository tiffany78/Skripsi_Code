Map.centerObject(bali_seaweed, 15.5);

// =====================
// PARAMETER UMUM
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var label = 'Class';

// =====================
// TRAINING POINTS
// =====================
var training = seaweed2.merge(nonseaweed2);
print('Training points:', training);

// =====================
// FUNGSI CLOUD MASK
// =====================
function maskCloud(img) {
  return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
}

// =====================
// FUNGSI MEMBANGUN KOMPOSIT
// =====================
function buildComposite(startDate, endDate, layerName, showLayer) {
  var collection = s2
    .filterDate(startDate, endDate)   // endDate eksklusif
    .filterBounds(bali2)
    .linkCollection(csPlus, [QA_BAND])
    .map(maskCloud);

  var count = collection.size();
  var composite = collection.median().clip(bali2);

  var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
  composite = composite.addBands(ndwi);

  var waterMask = composite.select('NDWI').gt(0);
  var compositeWater = composite.updateMask(waterMask);

  if (showLayer) {
    Map.addLayer(
      compositeWater,
      {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500},
      'RGB NDWI > 0 - ' + layerName,
      true
    );
  }

  return {
    image: compositeWater,
    count: count
  };
}

// =====================
// FUNGSI EKSPERIMEN 1 KANDIDAT
// =====================
var startDate = '2025-11-01';
var endDate = '2026-01-01';

var result = buildComposite(startDate, endDate, name, true);
var compositeWater = ee.Image(result.image);
var imageCount = ee.Number(result.count);

// Predictor bands
var bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'];
var input = compositeWater.select(bands);

// sampleRegions mengambil nilai piksel pada scale yang ditentukan
var samples = input.sampleRegions({
  collection: training,
  properties: [label],
  scale: 20,        // karena ada B8A
  geometries: true,
  tileScale: 4
});

var sampleCount = samples.size();

// Split
var withRandom = samples.randomColumn('random', 42);
var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet  = withRandom.filter(ee.Filter.gte('random', 0.7));

// RF baseline
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 50,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.7,
  seed: 42
}).train({
  features: trainSet,
  classProperty: label,
  inputProperties: bands
});

// Evaluasi test
var testClassification = testSet.classify(classifier);
var testCM = testClassification.errorMatrix(label, 'classification');

// Visual hasil klasifikasi kandidat
var classified = input.classify(classifier);
Map.addLayer(
  classified.clip(bali2),
  {min: 0, max: 1, palette: ['#d6d4a9', '#3a8a32']},
  'Prediction - ' + name,
  false
);

print('=== ' + name + ' ===');
print('Jumlah citra:', imageCount);
print('Jumlah sample:', sampleCount);
print('Confusion Matrix (Test):', testCM);
print('Test OA:', testCM.accuracy());
print('Recall / Producers Accuracy:', testCM.producersAccuracy());
print('Precision / Users Accuracy:', testCM.consumersAccuracy());

// =====================
// DAFTAR KANDIDAT KOMPOSIT
// =====================
var candidates = [
  // bulanan
  {name: 'April 2025', start: '2025-04-01', end: '2025-05-01'},
  {name: 'Mei 2025', start: '2025-05-01', end: '2025-06-01'},
  {name: 'Juli 2025', start: '2025-07-01', end: '2025-08-01'},
  {name: 'Aug 2025', start: '2025-08-01', end: '2025-09-01'},
  {name: 'Nov 2025', start: '2025-11-01', end: '2025-12-01'},
  {name: 'Des 2025', start: '2025-12-01', end: '2026-01-01'}, // best
  // 2 bulan 
  {name: 'April_Mei 2025', start: '2025-04-01', end: '2025-06-01'},
  {name: 'Juli_Aug 2025', start: '2025-07-01', end: '2025-09-01'},
  {name: 'Nov_Des 2025',  start: '2025-11-01', end: '2026-01-01'}, // best
  // 3 bulan 
  {name: 'Oct_Des 2025',  start: '2025-10-01', end: '2026-01-01'} //best
];

// =====================
// JALANKAN SEMUA EKSPERIMEN
// =====================
var experimentResults = ee.FeatureCollection(
  candidates.map(function(c) {
    return runExperiment(c.name, c.start, c.end);
  })
);

print('Ringkasan hasil eksperimen:', experimentResults);

// =====================
// URUTKAN HASIL TERBAIK
// =====================
var sortedResults = experimentResults.sort('testOA', false);
print('Hasil diurutkan berdasarkan Test OA:', sortedResults);

// =====================
// CHART
// =====================
var chartOA = ui.Chart.feature.byFeature(sortedResults, 'candidate', ['testOA'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Perbandingan Test OA Kandidat Komposit',
    hAxis: {title: 'Kandidat'},
    vAxis: {title: 'Overall Accuracy'},
    legend: {position: 'none'}
  });
print(chartOA);

var chartKappa = ui.Chart.feature.byFeature(sortedResults, 'candidate', ['testKappa'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Perbandingan Test Kappa Kandidat Komposit',
    hAxis: {title: 'Kandidat'},
    vAxis: {title: 'Kappa'},
    legend: {position: 'none'}
  });
print(chartKappa);


// =====================
// PILIH 1 KANDIDAT TERBAIK UNTUK TRAINING FINAL / EXPORT
// misalnya sementara dipilih manual setelah lihat hasil
// =====================
var bestComposite = buildComposite('2025-10-01', '2026-01-01', 'BEST Oct-Des 2025', true).image;

// =====================
// EXPORT DATA TRAINING DARI KANDIDAT TERPILIH
// =====================
var training = seaweed2.merge(nonseaweed2);
var scaled = bestComposite
  .select(['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9', 'B11', 'B12'])
  .multiply(0.0001);

var ndvi = scaled.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi2 = scaled.normalizedDifference(['B3', 'B8']).rename('NDWI');
var predictorImage = scaled.addBands([ndvi, ndwi2]);

var sampled = predictorImage.sampleRegions({
  collection: training,
  properties: ['Class'],
  scale: 20,
  geometries: true,
  tileScale: 4
});

// Tambahkan lon-lat supaya mudah di Python
var sampledWithXY = sampled.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    lon: coords.get(0),
    lat: coords.get(1)
  });
});

var exportCols = [
  'Class',
  'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9','B11', 'B12',
  'NDVI', 'NDWI',
  'lon', 'lat'
];

Export.table.toDrive({
  collection: sampledWithXY.select(exportCols),
  description: 'training_seaweed_Oct-Des_2025',
  folder: 'GEE_Export',
  fileFormat: 'CSV'
});

// =====================
// LAYER BATAS
// =====================
var batas50K = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Garis50K");
Map.addLayer(batas50K, {color: 'yellow', strokeWidth: 3}, 'batas50K');