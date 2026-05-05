// =====================
// ASMAT - PREDIKSI RUMPUT LAUT DENGAN GAP FILLING
// TANPA validMask, TETAPI TETAP MEMAKAI waterMask
// =====================

// =====================
// PARAMETER UTAMA
// =====================
var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// PENTING:
// Jika classifier asset dibuat dari alur training yang Anda upload,
// gunakan 'raw10000'.
// Jika classifier dilatih dari reflectance 0-1, ganti jadi 'reflectance01'.
var MODEL_INPUT_MODE = 'reflectance01';   // 'raw10000' atau 'reflectance01'

// Asset classifier
var savedClassifier = ee.Classifier.load(
  'projects/ee-tiffanytasyaagatha/assets/RF30_2_Bali_2025'
);

// Band common yang sama di S2, Landsat 8, Landsat 9

var commonBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

// Periode prediksi
var years = {
  q25_1: ['2025-01-01', '2025-04-01'],
  q25_2: ['2025-04-01', '2025-07-01'],
  q25_3: ['2025-07-01', '2025-10-01'],
  q25_4: ['2025-10-01', '2026-01-01']
};

// =====================
// DATASET
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");

// =====================
// VISUAL
// =====================
var rgbVisRaw = {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500};
var rgbVisRefl = {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.20};

var classificationVis = {
  min: 0,
  max: 1,
  palette: ['#d6d4a9', '#3a8a32']
};

var probVis = {
  min: 0,
  max: 1,
  palette: ['#f7f7f7', '#d9f0d3', '#78c679', '#238443']
};

// =====================
// FUNGSI HELPER
// =====================

// normalizedDifference() di GEE memask bila ada nilai negatif.
// Jadi untuk Landsat gunakan expression.
function calcNDWI(image, greenBand, nirBand) {
  return image.expression(
    '(g - n) / (g + n)',
    {
      g: image.select(greenBand),
      n: image.select(nirBand)
    }
  ).rename('NDWI');
}

function choosePredictorScale(rawImage, reflImage) {
  return ee.Image(MODEL_INPUT_MODE === 'raw10000' ? rawImage : reflImage);
}

// =====================
// PREP SENTINEL-2
// =====================
function prepS2(img) {
  var clear = img.select(QA_BAND).gte(CLEAR_THRESHOLD);

  // raw domain sesuai training lama
  var raw = img
    .updateMask(clear)
    .select(commonBands);

  // reflectance 0-1
  var refl = raw.multiply(0.0001);

  var predictor = choosePredictorScale(raw, refl);
  var ndwi = calcNDWI(refl, 'B3', 'B8');

  return predictor
    .addBands(ndwi)
    .copyProperties(img, img.propertyNames());
}

// =====================
// PREP LANDSAT 8/9
// =====================
function prepLandsat(img) {
  var qa = img.select('QA_PIXEL');
  var radSat = img.select('QA_RADSAT').eq(0);

  var clear = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))       // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0))       // snow
    .and(radSat);

  // reflectance Landsat 0-1
  var refl = img
    .updateMask(clear)
    .select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
      ['B2',    'B3',    'B4',    'B8',    'B11',   'B12']
    )
    .multiply(0.0000275)
    .add(-0.2);

  // agar bisa cocok dengan model lama berbasis raw S2 ~ 0..10000
  // negative reflectance dipotong ke 0 supaya tidak bikin domain aneh
  var rawLikeS2 = refl.max(0).multiply(10000);

  var predictor = choosePredictorScale(rawLikeS2, refl);
  var ndwi = calcNDWI(refl, 'B3', 'B8');

  return predictor
    .addBands(ndwi)
    .copyProperties(img, img.propertyNames());
}

// =====================
// LOAD COLLECTIONS
// =====================
function getS2Collection(startDate, endDate) {
  return s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(prepS2);
}

function getLandsatCollection(startDate, endDate) {
  var col8 = l8
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .map(prepLandsat);

  var col9 = l9
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .map(prepLandsat);

  // default reprojection di EE adalah nearest-neighbor,
  // jadi kita haluskan input Landsat untuk tampilan/proyeksi lintas grid
  return col8.merge(col9).map(function(img) {
    return img.resample('bilinear');
  });
}

// =====================
// BUILD GAP-FILLED PREDICTOR
// =====================
function buildGapFilledPredictor(startDate, endDate, layerName) {
  var start = ee.Date(startDate);
  var end = ee.Date(endDate);

  // wider window untuk backup
  var startWide = start.advance(-1, 'month');
  var endWide = end.advance(1, 'month');

  var s2Exact = getS2Collection(start, end);
  var s2Wide  = getS2Collection(startWide, endWide);
  var lsWide  = getLandsatCollection(startWide, endWide);

  // print('Jumlah S2 exact - ' + layerName, s2Exact.size());
  // print('Jumlah S2 wide - ' + layerName, s2Wide.size());
  // print('Jumlah Landsat wide - ' + layerName, lsWide.size());

  // Water mask TETAP dipakai walau validMask diabaikan
  var waterMask = s2Wide.select('NDWI')
    .merge(lsWide.select('NDWI'))
    .median()
    .gt(0)
    .clip(buffer);

  // Komposit utama
  var s2Primary = s2Exact.select(commonBands).median()
    .updateMask(waterMask)
    .clip(buffer);

  var s2Backup = s2Wide.select(commonBands).median()
    .updateMask(waterMask)
    .clip(buffer);

  var lsBackup = lsWide.select(commonBands).median()
    .updateMask(waterMask)
    .clip(buffer);

  // Gap filling
  var gapFilled = s2Primary
    .unmask(s2Backup, false)
    .unmask(lsBackup, false)
    .updateMask(waterMask)
    .clip(buffer);

  // Di area mana Landsat benar-benar dipakai?
  var filledByLandsat = s2Primary.select('B2').mask().not()
    .and(s2Backup.select('B2').mask().not())
    .and(lsBackup.select('B2').mask())
    .selfMask()
    .clip(buffer);

  return {
    predictor: gapFilled,
    waterMask: waterMask,
    filledByLandsat: filledByLandsat
  };
}

// =====================
// DIAGNOSTIK
// =====================
function printBandRanges(image, layerName) {
  var stats = image.reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: buffer.geometry(),
    scale: 30,
    maxPixels: 1e13,
    tileScale: 4
  });
  print('Band min/max - ' + layerName, stats);
}

function printClassHistogram(classified, layerName) {
  var hist = classified.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: buffer.geometry(),
    scale: 30,
    maxPixels: 1e13,
    tileScale: 4
  });
  // print('Class histogram - ' + layerName, hist);
}

// =====================
// PREDIKSI PER PERIODE
// =====================
Object.keys(years).forEach(function(y) {
  var startDate = years[y][0];
  var endDate = years[y][1];

  var result = buildGapFilledPredictor(startDate, endDate, y);
  var input = result.predictor.select(commonBands);

  // printBandRanges(input, y);

  var classified = input.classify(savedClassifier).rename('class');
  // printClassHistogram(classified, y);

  Map.addLayer(
    input,
    MODEL_INPUT_MODE === 'raw10000' ? rgbVisRaw : rgbVisRefl,
    'GapFilled RGB - ' + y,
    true
  );

  Map.addLayer(
    result.waterMask.selfMask(),
    {palette: ['#00ffff']},
    'Water Mask - ' + y,
    false
  );

  Map.addLayer(
    result.filledByLandsat,
    {palette: ['#ffff00']},
    'Filled by Landsat - ' + y,
    false
  );

  // HAPUS layer probability dulu
  Map.addLayer(
    classified.updateMask(result.waterMask),
    classificationVis,
    'Prediction - ' + y,
    true
  );
});

// =====================
// BATAS
// =====================
var batas50K = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Garis50K");
Map.addLayer(batas50K, {color: 'yellow', strokeWidth: 2}, 'batas50K', true);