var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

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
// VISUAL PARAMETER
// =====================
var ndwiWaterVis = {
  min: 0,
  max: 1,
  palette: [
    '#f7fcfd',
    '#CCE4EC',
    '#66B7C2',
    '#236F8B',
    '#002944'
  ]
};

var ndtiClassVis = {
  min: 1,
  max: 5,
  palette: [
    '#2166ac', // 1
    '#67a9cf', // 2
    '#d9ef8b', // 3
    '#fdae61', // 4
    '#d73027'  // 5
  ]
};

// =====================
// PREP FUNCTION
// =====================

// Sentinel-2: mask clear pixel dari Cloud Score+
function prepS2(img) {
  var clear = img.select(QA_BAND).gte(CLEAR_THRESHOLD);

  var refl = img
    .updateMask(clear)
    .select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'nir'])
    .divide(10000);

  var ndwi = refl.normalizedDifference(['green', 'nir']).rename('NDWI');

  return refl.addBands(ndwi)
             .copyProperties(img, img.propertyNames());
}

// Landsat 8/9: mask awan/bayangan/cirrus/snow dari QA_PIXEL
function prepLandsat(img) {
  var qa = img.select('QA_PIXEL');

  var clear = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))       // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0));      // snow

  var refl = img
    .updateMask(clear)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'], ['blue', 'green', 'red', 'nir'])
    .multiply(0.0000275)
    .add(-0.2);

  var ndwi = refl.normalizedDifference(['green', 'nir']).rename('NDWI');

  return refl.addBands(ndwi)
             .copyProperties(img, img.propertyNames());
}

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

  return col8.merge(col9);
}

// =====================
// GAP FILL FUNCTION
// =====================
// Prioritas isi:
// 1) Sentinel periode utama
// 2) Sentinel periode diperlebar
// 3) Landsat periode diperlebar
function buildGapFilledComposite(startDate, endDate, layerName) {
  var start = ee.Date(startDate);
  var end = ee.Date(endDate);

  // window diperlebar 1 bulan ke kiri-kanan
  var startWide = start.advance(-1, 'month');
  var endWide = end.advance(1, 'month');

  var s2Exact = getS2Collection(start, end);
  var s2Wide  = getS2Collection(startWide, endWide);
  var lsWide  = getLandsatCollection(startWide, endWide);

  // print('Jumlah image S2 exact - ' + layerName, s2Exact.size());
  // print('Jumlah image S2 wide - ' + layerName, s2Wide.size());
  // print('Jumlah image Landsat wide - ' + layerName, lsWide.size());

  // Water mask dibuat dari gabungan NDWI S2 wide + Landsat wide
  var waterMask = s2Wide.select('NDWI')
    .merge(lsWide.select('NDWI'))
    .median()
    .gt(0)
    .clip(buffer)
    .rename('water_mask');

  var s2Primary = s2Exact
    .select(['blue', 'green', 'red', 'nir'])
    .median()
    .updateMask(waterMask)
    .clip(buffer);

  var s2Backup = s2Wide
    .select(['blue', 'green', 'red', 'nir'])
    .median()
    .updateMask(waterMask)
    .clip(buffer);

  var lsBackup = lsWide
    .select(['blue', 'green', 'red', 'nir'])
    .median()
    .updateMask(waterMask)
    .clip(buffer)
    .resample('bilinear');

  var gapFilled = s2Primary
    .unmask(s2Backup, false)
    .unmask(lsBackup, false)
    .clip(buffer);

  // Diagnostik: area yang benar-benar terisi oleh Landsat
  var filledByLandsat = s2Primary.select('blue').mask().not()
    .and(s2Backup.select('blue').mask().not())
    .and(lsBackup.select('blue').mask())
    .selfMask()
    .clip(buffer);

  return {
    image: gapFilled,
    waterMask: waterMask,
    filledByLandsat: filledByLandsat
  };
}

// =====================
// MAIN PERIOD FUNCTION
// =====================
function predictPeriod(startDate, endDate, layerName, showLayer) {
  var result = buildGapFilledComposite(startDate, endDate, layerName);
  var composite = result.image;

  Map.addLayer(
    composite,
    {bands: ['red', 'green', 'blue'], min: 0.02, max: 0.15},
    'RGB GapFilled - ' + layerName,
    false
  );

  Map.addLayer(
    result.filledByLandsat,
    {palette: ['#ffff00']},
    'Filled by Landsat - ' + layerName,
    false
  );

  // ---------------------
  // NDWI
  // ---------------------
  var ndwi = composite.normalizedDifference(['green', 'nir'])
    .rename('NDWI')
    .clamp(-1, 1);

  // final water mask: gabungan waterMask awal + NDWI hasil gap-filled
  var waterMask = result.waterMask.and(ndwi.gt(0)).rename('water_mask_final');

  var ndwiWater = ndwi.updateMask(waterMask);

  Map.addLayer(
    ndwiWater,
    ndwiWaterVis,
    'NDWI Water Only - ' + layerName,
    showLayer
  );

  // ---------------------
  // NDTI
  // ---------------------
  var ndti = composite.normalizedDifference(['red', 'green'])
    .rename('NDTI')
    .clamp(-1, 1);

  var ndtiWater = ndti.updateMask(waterMask);

  var ndtiClass = ee.Image(0)
    .where(ndtiWater.lt(0), 1)
    .where(ndtiWater.gte(0).and(ndtiWater.lt(0.1)), 2)
    .where(ndtiWater.gte(0.1).and(ndtiWater.lt(0.2)), 3)
    .where(ndtiWater.gte(0.2).and(ndtiWater.lt(0.25)), 4)
    .where(ndtiWater.gte(0.25), 5)
    .updateMask(waterMask)
    .rename('NDTI_Class');

  Map.addLayer(
    ndtiClass,
    ndtiClassVis,
    'NDTI Classes - ' + layerName,
    showLayer
  );

  return {
    ndtiClass: ndtiClass,
    waterMask: waterMask,
    composite: composite
  };
}

// =====================
// RUN PER QUARTER
// =====================
var q1 = predictPeriod('2025-01-01', '2025-04-01', 'Q1-2025', false);
var q2 = predictPeriod('2025-04-01', '2025-07-01', 'Q2-2025', false);
var q3 = predictPeriod('2025-07-01', '2025-10-01', 'Q3-2025', false);
var q4 = predictPeriod('2025-10-01', '2026-01-01', 'Q4-2025', false);

// =====================
// WATER FREQUENCY DARI HASIL GAP-FILLED
// =====================
// gte(2) = dianggap air stabil minimal muncul di 2 quarter
// kalau terlalu ketat, bisa diganti gte(1)
var waterFreq = q1.waterMask.unmask(0)
  .add(q2.waterMask.unmask(0))
  .add(q3.waterMask.unmask(0))
  .add(q4.waterMask.unmask(0))
  .rename('Water_Frequency');

var stableWaterMask = waterFreq.gte(2).clip(buffer);

Map.addLayer(
  stableWaterMask.selfMask(),
  {palette: ['#00c8ff']},
  'Stable Water Mask',
  false
);

// =====================
// HIGH TURBIDITY
// =====================
var highSedFreq = q1.ndtiClass.gte(2).unmask(0)
  .add(q2.ndtiClass.gte(2).unmask(0))
  .add(q3.ndtiClass.gte(2).unmask(0))
  .add(q4.ndtiClass.gte(2).unmask(0))
  .rename('HighSed_Frequency');

var highSedMask = highSedFreq.gt(1).clip(buffer);
var lowSedMask  = highSedFreq.lte(1).clip(buffer);

Map.addLayer(
  highSedMask.selfMask(),
  {palette: ['#d94701']},
  'Excluded High Sediment',
  true
);

// =====================
// BATNAS SEBAGAI TARGET DEPTH
// =====================
var batnasRaw = ee.Image("projects/ee-tiffanytasyaagatha/assets/Batnas");

// kedalaman positif (opsional untuk visualisasi/analisis lanjut)
var batnasDepth = batnasRaw.multiply(-1).rename('depth');

// batasi range kedalaman
var minDepth = 0;
var maxDepth = 25;

var depthMask = batnasRaw.lte(-minDepth).and(batnasRaw.gte(-maxDepth));

// =====================
// MASK FINAL UNTUK ANALISIS
// hanya air stabil + low sediment + depth valid
// =====================
var validMask = stableWaterMask
  .and(lowSedMask)
  .and(depthMask)
  .clip(buffer);

Map.addLayer(
  validMask.selfMask(),
  {palette: ['#00ff00']},
  'Valid Area for Bathymetry RF',
  true
);

// ubah jadi 0/1
var validMaskAssetImage = validMask
  .unmask(0)
  .toByte()
  .rename('valid_mask');

Map.addLayer(
  validMaskAssetImage.updateMask(validMaskAssetImage),
  {palette: ['#00ff00']},
  'validMask for export',
  true
);

// ekspor ke Asset
Export.image.toAsset({
  image: validMaskAssetImage,
  description: 'export_validMask_bathymetry_gapfilled_2025',
  assetId: 'projects/ee-tiffanytasyaagatha/assets/validMask_bathymetry_gapfilled_2025',
  region: buffer.geometry(),
  scale: 20,
  maxPixels: 1e13,
  pyramidingPolicy: {'.default': 'sample'}
});