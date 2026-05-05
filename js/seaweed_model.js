Map.centerObject(bali_seaweed, 15.5);

// Batimetri Nasional
// var batnas = ee.Image("projects/ee-tiffanytasyaagatha/assets/Bali_Batnas2");
// var lautOnly = batnas.updateMask(batnas.lte(1));
// Map.addLayer(
//   lautOnly.clip(bali2),
//   {
//     min: -127,
//     max: 1,
//     palette: ['#001219','#005f73','#0a9396','#94d2bd','#e9d8a6']
//   },
//   'BATNAS Bathymetry'
// );

// VISUALUSASI CITRA
var csPlus=ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var composite = s2
    .filterDate("2025-01-01", "2025-12-31")
    .filterBounds(bali2)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));})
    ;
composite = composite.median();
Map.addLayer(
    composite.clip(bali2), 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    'RGB Median');
    
// NDWI 
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
composite = composite.addBands(ndwi);

// Mask hanya NDWI > 0
var waterMask = composite.select('NDWI').gt(0);
var composite_water = composite.updateMask(waterMask);
Map.addLayer(
  composite_water.clip(bali2),
  {bands: ['B4','B3','B2'], min:0, max:2500},
  'RGB NDWI > 0'
);

// MODEL RANDOM FOREST
var training = seaweed.merge(non_seaweed);
print(training);

var label = "Class";
var bands = ['B2', 'B3', 'B4', 'B8A'];
var input = composite_water.select(bands);

var trainImage = input.sampleRegions({
  collection: training,
  properties: [label],
  scale: 10
});
print(trainImage);

var trainingData = trainImage.randomColumn()
var trainSet = trainingData.filter(ee.Filter.lessThan("random", 0.8));
var testSet = trainingData.filter(ee.Filter.greaterThanOrEquals("random", 0.8));

// Model training
var classifier = ee.Classifier.smileRandomForest(50)
  .train({
    features: trainSet,
    classProperty: label,
    inputProperties: bands
  });
  
// classify image
var classified = input.classify(classifier);
var classificationVis = {
  min: 0, 
  max: 1, 
  palette: ['#d6d4a9', '#3a8a32']
};
Map.addLayer(classified.clip(bali2), classificationVis, "Prediction");

// TestSet untuk evaluasi
var testClassification = testSet.classify(classifier);
// confusion matrix di test data
var testConfusionMatrix = testClassification.errorMatrix(label, 'classification');
print('Confusion Matrix (Test):', testConfusionMatrix);
print('Test overall accuracy:', testConfusionMatrix.accuracy());
print('Producers Accuracy (Recall):', testConfusionMatrix.producersAccuracy());
print('Users Accuracy (Precision):', testConfusionMatrix.consumersAccuracy());

// Pemanggilan Asset
var batas50K = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Garis50K");
// Pembuatan Layer
Map.addLayer(batas50K, {color: 'yellow', strokeWidth : 10},'batas50K');

// ASMAT 
// Pemanggilan Asset
var batas = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Batas2024");
// Print Atribut Data
print(batas.limit(5));
// Filtering
var asmat = batas.filter(
  ee.Filter.stringContains('WADMKK', 'Asmat')
);
// Pembuatan Layer
Map.addLayer(asmat, {color: 'blue', strokeWidth : 10},'batas24');
Map.centerObject(asmat, 8);

// PEMBUATAN AOI
var garisPantai = ee.Geometry.LineString([
  batasAtas.coordinates(),
  batasBawah.coordinates()
]);
var jarak12Mil = 12 * 1852; // meter
var zonaPesisir = garisPantai.buffer(jarak12Mil);

// Map.addLayer(garisPantai, {color: 'red'}, 'Garis Pantai');
// Map.addLayer(zonaPesisir, {color: 'green'}, 'Zona 12 Mil Laut');

// ROI 12 MIL
var garisPantaiAsmat = batas50K.filterBounds(asmat);
var coastlineAll = garisPantaiAsmat.geometry().dissolve();
var coastline = coastlineAll.intersection(
  asmat.geometry().buffer(100),
  ee.ErrorMargin(1)
);
Map.addLayer(coastline, {color:'red'}, 'Coastline Asmat FIX');

var panjang = coastline.length();
var interval = 1000;
var distances = ee.List.sequence(0, panjang, interval);

var segments = coastline.cutLines(distances);
// Map.addLayer(segments, {color:'purple'}, 'Segments 1km');

var titikSampling = ee.FeatureCollection(
  segments.geometries().map(function(g) {
    return ee.Feature(ee.Geometry(g).centroid());
  })
);
Map.addLayer(titikSampling, {color:'red'}, 'Titik 1km');

var jarak12Mil = 12 * 1852; // meter
var buffer12Mil = titikSampling.map(function(f) {
  return f.buffer(jarak12Mil);
});
Map.addLayer(buffer12Mil, {color:'green'}, 'Buffer 12 Mil');

// PREDICT ASMAT
var calculateNDWI = function(image){
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename("NDWI");
  return image.addBands(ndwi);
};

var classificationVis = {
  min: 0, 
  max: 1, 
  palette: ['#d6d4a9', '#3a8a32']
};

// DEFINE YEARS
var years = {
  y1: ['2022-01-01', '2022-12-01'],
  y2: ['2023-01-01', '2023-12-01'],
  y3: ['2024-01-01', '2024-12-01'],
  y4: ['2024-01-01', '2025-12-01']
};

// LOOP THROUGH YEARS
Object.keys(years).forEach(function(q){
  
  var startDate = years[q][0];
  var endDate   = years[q][1];
  
  // Step 1: Load and prepare image
  var newImage = s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer12Mil)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));})
    .map(calculateNDWI)
    .median();
    
  // print("i: ", q, " ", newImage);
  
  var newImageCropped = newImage.clip(buffer12Mil);
  Map.addLayer(newImageCropped, 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    "True Color " + q);
  
  // Step 2: Select bands (same as training)
  var newImageInput = newImageCropped.select(bands);
  
  // Step 3: Classify using existing classifier
  var newClassification = newImageInput.classify(classifier);
  Map.addLayer(newClassification, classificationVis, "Prediction " + q);
});

// VISUALUSASI CITRA
var csPlus=ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var composite = s2
    .filterDate("2025-01-01", "2025-12-31")
    .filterBounds(buffer12Mil)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));})
    ;
composite = composite.median().clip(buffer12Mil);

Map.addLayer(
    composite, 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    'RGB Median');
    
// NDWI 
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
composite = composite.addBands(ndwi);

// Mask hanya NDWI > 0
var waterMask = composite.select('NDWI').gt(0);
var composite_water = composite.updateMask(waterMask);
Map.addLayer(
  composite_water.clip(buffer12Mil),
  {bands: ['B4','B3','B2'], min:0, max:2500},
  'RGB NDWI > 0'
);

// classify image
var bands = ['B2', 'B3', 'B4', 'B8A'];
var input = composite_water.select(bands);
var classified = input.classify(classifier);
var classificationVis = {
  min: 0, 
  max: 1, 
  palette: ['#d6d4a9', '#3a8a32']
};
Map.addLayer(classified.clip(buffer12Mil), classificationVis, "Prediction Asmat");