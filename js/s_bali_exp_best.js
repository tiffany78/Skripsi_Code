// ASMAT
var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// VALID IMAGE
var validMaskAsset = ee.Image('projects/ee-tiffanytasyaagatha/assets/validMask2_2025')
  .select('valid_mask')
  .eq(1);

var validImage = validMaskAsset.updateMask(validMaskAsset);

// CITRA
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// CLASSIFIER
var bestBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B11', 'B12'];
var savedClassifier = ee.Classifier.load(
  'projects/ee-tiffanytasyaagatha/assets/RF_Bali_2025'
);

var classificationVis = {
  min: 0,
  max: 1,
  palette: ['#d6d4a9', '#3a8a32']
};

var years = {
  q25_1: ['2025-01-01', '2025-04-01'],
  q25_2: ['2025-04-01', '2025-07-01'],
  q25_3: ['2025-07-01', '2025-10-01'],
  q25_4: ['2025-10-01', '2026-01-01']
};

function calculateNDWI(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename("NDWI");
  return image.addBands(ndwi);
}

Object.keys(years).forEach(function(y) {
  var startDate = years[y][0];
  var endDate = years[y][1];

  var newImage = s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    })
    .map(calculateNDWI)
    .median()
    .clip(buffer);

  // NDWI > 0
  var waterMask = newImage.select('NDWI').gt(0);

  // gabungkan dengan validMaskAsset
  var finalMask = waterMask.and(validMaskAsset);

  // image yang hanya valid di area air
  var newImageMasked = newImage.updateMask(finalMask);

  // input klasifikasi (filter water + valid)
  var newImageInput = newImageMasked.select(bestBands);
  // klasifikasi
  var newClassification = newImageInput.classify(savedClassifier)
                                      .updateMask(finalMask);

  // input (raw)
  // var newImageInput = newImage.select(bestBands);
  // klasifikasi
  // var newClassification = newImageInput.classify(savedClassifier);
                                      
  Map.addLayer(
    newImageMasked,
    {bands: ['B4','B3','B2'], min:0, max:2500},
    "True Color " + y,
    false
  );

  Map.addLayer(
    newClassification,
    classificationVis,
    "Prediction " + y,
    true
  );
});