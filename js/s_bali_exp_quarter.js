// =====================
// ROI
// =====================
// Ganti trainROI jika ROI training Bali Anda bukan bali2.
var trainROI = bali2;

// ROI buffer Asmat
var asmatBuffer = ee.FeatureCollection(
  "projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat"
);
var asmatROI = asmatBuffer.geometry();

// Fokus tampilan ke Asmat karena hasil prediksi utama ada di sana
Map.centerObject(asmatBuffer, 8);
Map.centerObject(bali2, 13);

// =====================
// PARAMETER UMUM
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var label = 'Class';

// Band prediktor
var bands = ['B2', 'B3', 'B4', 'B8'];

// Visualisasi klasifikasi
var classVis = {
  min: 0,
  max: 1,
  palette: ['white', 'red']
};

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
// FUNGSI MEMBANGUN KOMPOSIT 0-1
// =====================
function buildComposite01(startDate, endDate, roi, layerName, showLayer) {
  var collection = s2
    .filterDate(startDate, endDate)   // endDate eksklusif
    .filterBounds(roi)
    .linkCollection(csPlus, [QA_BAND])
    .map(maskCloud);

  var imageCount = collection.size();

  // Median komposit dan scale
  var img = ee.Image(collection.median()).clip(roi).multiply(0.0001);

  // NDWI dihitung dari citra yang sudah di-scale
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var composite = img
    .addBands(ndwi)
    .select(bands)
    .rename(bands);

  Map.addLayer(
    composite,
    {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3},
    'RGB - ' + layerName,
    showLayer
  );

  return {
    image: composite,
    imageCount: imageCount
  };
}

// =====================
// FUNGSI HITUNG LUAS KELAS (ha)
// =====================
function calcClassAreaHa(classifiedImage, roi, classValue) {
  var areaImage = ee.Image.pixelArea()
    .divide(10000) // m2 -> ha
    .rename('area')
    .updateMask(classifiedImage.eq(classValue));

  var area = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 20,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 8
  }).get('area');

  // jaga-jaga kalau null
  return ee.Number(ee.Algorithms.If(area, area, 0));
}

// =====================
// FUNGSI EKSPERIMEN PER QUARTER
// =====================
function runExperiment(name, startDate, endDate) {
  var trainResult = buildComposite01(
    startDate, endDate, trainROI,
    'Bali ' + name,
    false
  );

  var compositeTrain = ee.Image(trainResult.image);
  var inputTrain = compositeTrain.select(bands);

  var samples = inputTrain.sampleRegions({
    collection: training,
    properties: [label],
    scale: 10,
    geometries: true,
    tileScale: 4
  }).filter(ee.Filter.notNull(bands));

  var sampleCount = samples.size();

  var withRandom = samples.randomColumn('random', 42);
  var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
  var testSet  = withRandom.filter(ee.Filter.gte('random', 0.7));

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 80,
    variablesPerSplit: null,
    minLeafPopulation: 2,
    bagFraction: 0.6,
    seed: 42
  }).train({
    features: trainSet,
    classProperty: label,
    inputProperties: bands
  });

  var testClassification = testSet.classify(classifier);
  var testCM = testClassification.errorMatrix(label, 'classification');

  // Asmat quarter yang sama
  var asmatResult = buildComposite01(
    startDate, endDate, asmatROI,
    'Asmat ' + name,
    false
  );

  var compositeAsmat = ee.Image(asmatResult.image);
  var inputAsmat = compositeAsmat.select(bands);

  var predAsmatNoNDWI = inputAsmat
    .classify(classifier)
    .clip(asmatROI)
    .rename('classification');

  var waterMask = compositeAsmat.select('NDWI').gt(0);

  var predAsmatWithNDWI = predAsmatNoNDWI
    .updateMask(waterMask)
    .rename('classification');

  // kelas rumput laut = 1
  var seaweedClass = 1;

  var areaNoNDWI_ha = calcClassAreaHa(predAsmatNoNDWI, asmatROI, seaweedClass);
  var areaWithNDWI_ha = calcClassAreaHa(predAsmatWithNDWI, asmatROI, seaweedClass);

  // opsional: tetap tampilkan layer
  Map.addLayer(predAsmatNoNDWI, classVis, 'Asmat NO NDWI - ' + name, false);
  Map.addLayer(predAsmatWithNDWI, classVis, 'Asmat WITH NDWI - ' + name, false);
  
  // print
  print('=== ' + name + ' ===');
  print('Test OA:', testCM.accuracy());
  print('Test Kappa:', testCM.kappa());
  print('Recall / Producers Accuracy:', testCM.producersAccuracy());
  print('Precision / Users Accuracy:', testCM.consumersAccuracy());

  return ee.Feature(null, {
    candidate: name,
    startDate: startDate,
    endDate: endDate,
    baliImageCount: trainResult.imageCount,
    asmatImageCount: asmatResult.imageCount,
    sampleCount: sampleCount,
    trainCount: trainSet.size(),
    testCount: testSet.size(),
    testOA: testCM.accuracy(),
    testKappa: testCM.kappa(),
    asmatSeaweedAreaNoNDWI_ha: areaNoNDWI_ha,
    asmatSeaweedAreaWithNDWI_ha: areaWithNDWI_ha
  });
}

// =====================
// DAFTAR KANDIDAT KOMPOSIT
// =====================
var candidates = [
  {name: 'Jan_Mar 2025',   start: '2025-01-01', end: '2025-04-01'},
  {name: 'Apr_Jun 2025',   start: '2025-04-01', end: '2025-07-01'},
  {name: 'July_Sept 2025', start: '2025-07-01', end: '2025-10-01'},
  {name: 'Oct_Des 2025',   start: '2025-10-01', end: '2026-01-01'}
];

// var candidates = [
//   {name: '2022',   start: '2022-01-01', end: '2023-01-01'},
//   {name: '2023',   start: '2023-01-01', end: '2024-01-01'},
//   {name: '2024', start: '2024-01-01', end: '2025-01-01'},
//   {name: '2025',   start: '2025-01-01', end: '2026-01-01'}
// ];

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
// URUTKAN HASIL
// =====================
var sortedResults = experimentResults.sort('testOA', false);
// print('Hasil diurutkan berdasarkan Test OA:', sortedResults);

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
// EXPORT CSV HASIL RINGKASAN
// =====================
Export.table.toDrive({
  collection: experimentResults,
  description: 'Asmat_Seaweed_Quarter_Area_Summary',
  folder: 'GEE_Export',
  fileNamePrefix: 'asmat_seaweed_quarter_area_summary',
  fileFormat: 'CSV'
});