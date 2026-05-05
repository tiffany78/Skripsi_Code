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

function buildPredictors(startDate, endDate, maxDepth, model) {
  var composite = buildComposite(startDate, endDate);

  var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var waterMask = ndwi.gt(0);

  var shallowMask = gebco2025.lt(0).and(gebco2025.gte(-maxDepth));

  var blue  = composite.select('B2').multiply(0.0001)
    .updateMask(waterMask)
    .updateMask(shallowMask);
  var green = composite.select('B3').multiply(0.0001)
    .updateMask(waterMask)
    .updateMask(shallowMask);
  var red   = composite.select('B4').multiply(0.0001)
    .updateMask(waterMask)
    .updateMask(shallowMask);
  var nir   = composite.select('B8').multiply(0.0001)
    .updateMask(waterMask)
    .updateMask(shallowMask);
  
  var predictors = ee.Image.cat([
    blue.rename('B2'),
    green.rename('B3'),
    red.rename('B4'),
    nir.rename('B8')
  ]).toFloat();

  var depth_est = predictors
    .classify(savedRF)
    .rename('Depth_RF')
    .updateMask(shallowMask)
    .max(0)
    .min(maxDepth);
  var depth_class = depthClass(depth_est);

  Map.addLayer(
  depth_class,
  depthClassVis,
  'Prediksi ' + ' (' + startDate + ' to ' + endDate + ')',
  true
  );
}

// =====================
// KONFIGURASI VISUALISASI
// =====================
function depthClass(depth_est){
    var depth_vis_class = ee.Image(0)
        .where(depth_est.gte(1).and(depth_est.lt(2)), 1)
        .where(depth_est.gte(2).and(depth_est.lt(3)), 2)
        .where(depth_est.gte(3).and(depth_est.lt(4)), 3)
        .where(depth_est.gte(4).and(depth_est.lt(5)), 4)
        .where(depth_est.gte(5).and(depth_est.lt(6)), 5)
        .where(depth_est.gte(6).and(depth_est.lt(7)), 6)
        .where(depth_est.gte(7).and(depth_est.lt(8)), 7)
        .where(depth_est.gte(8).and(depth_est.lt(9)), 8)
        .where(depth_est.gte(9).and(depth_est.lt(10)), 9)
        .where(depth_est.gte(10), 10)
        .updateMask(depth_est.mask())
        .rename('Depth_RF_Class');

    return depth_vis_class;
}

var depthClassVis = {
  min: 0,
  max: 10,
  palette: [
    '#f7fbff', // 0–<1
    '#deebf7', // 1–<2
    '#c6dbef', // 2–<3
    '#9ecae1', // 3–<4
    '#6baed6', // 4–<5
    '#4292c6', // 5–<6
    '#2171b5', // 6–<7
    '#08519c', // 7–<8
    '#08306b', // 8–<9
    '#041f4a', // 9–<10
    '#000000'  // >=10
  ]
};

// =====================
// LOAD MODEL KEDALAMAN 35
// =====================
var startDate = '2025-01-01';
var endDate = '2026-01-01';
var maxDepth = 35;

// BATNAS
var savedRF = ee.Classifier.load(
  'projects/ee-tiffanytasyaagatha/assets/depth_batnas_2025_10'
);
var savedMetrics = ee.FeatureCollection(
  "projects/ee-tiffanytasyaagatha/assets/depth_batnas_2025_10_doc"
);
var firstMetric = ee.Feature(savedMetrics.first());
print(
  'Properties:',
  firstMetric.toDictionary(['model_name', 'r2', 'rmse', 'mae'])
);
// Call Function
buildPredictors (startDate, endDate, maxDepth, savedRF);