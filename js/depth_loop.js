var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// =====================
// PARAMETER UMUM
// =====================
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// GEBCO
var gebco2025 = ee.Image("projects/ee-tiffanytasyaagatha/assets/gebco2025");

// RF params
var numScale = 20;
var numPixel = 250;
var rfParams = {
  numberOfTrees: 50,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.7,
  seed: 42
};

// VISUALISASI
var startDate = "2025-01-01";
var endDate   = "2026-01-01";

var composite = s2
  .filterDate(startDate, endDate)
  .filterBounds(buffer)
  .linkCollection(csPlus, [QA_BAND])
  .map(function(img) {
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
  })
  .median()
  .clip(buffer);

// --- water mask (NDWI)
var ndwi = composite.normalizedDifference(['B3','B8']);
var waterMask = ndwi.gt(0);
var img_water = composite.updateMask(waterMask);

// --- predictors (ubah ke reflectance 0-1)
// RF tidak wajib normalisasi lain; cukup scaling 0.0001 agar konsisten/rapi
var predictors = img_water
  .select(['B2','B3','B4','B8'])
  .multiply(0.0001)
  .toFloat();

// --- depth training (GEBCO: negatif -> kedalaman positif)
var depth = gebco2025
  .updateMask(gebco2025.lt(0))
  .multiply(-1)
  .rename('depth')
  .resample('bilinear')
  .updateMask(waterMask)
  .toFloat();

// (opsional tapi bagus) samakan proyeksi/resolusi depth ke 10 m
depth = depth.reproject({
  crs: predictors.select('B2').projection(),
  scale: 20
});

// =====================
// BANK FITUR
// =====================
var eps = 1e-6;

var lnBands = predictors.add(eps).log()
  .rename(['ln_B2','ln_B3','ln_B4','ln_B8']);

var r_B2B3 = predictors.select('B2').divide(predictors.select('B3').add(eps)).rename('r_B2_B3');
var r_B2B4 = predictors.select('B2').divide(predictors.select('B4').add(eps)).rename('r_B2_B4');
var r_B3B4 = predictors.select('B3').divide(predictors.select('B4').add(eps)).rename('r_B3_B4');

var B2 = predictors.select('B2').add(eps);
var B3 = predictors.select('B3').add(eps);
var B4 = predictors.select('B4').add(eps);

var lr_B2B3 = B2.log().divide(B3.log()).rename('lr_ln_B2_B3');
var lr_B2B4 = B2.log().divide(B4.log()).rename('lr_ln_B2_B4');

var featureBank = predictors
  .addBands(lnBands)
  .addBands([r_B2B3, r_B2B4, r_B3B4, lr_B2B3, lr_B2B4])
  .updateMask(waterMask);

var featureSets = [
  {name: 'S1_Base', bands: ['B2','B3','B4','B8']},
  {name: 'S2_Base_ln', bands: ['B2','B3','B4','B8','ln_B2','ln_B3','ln_B4','ln_B8']},
  {name: 'S3_Base_ratio', bands: ['B2','B3','B4','B8','r_B2_B3','r_B2_B4','r_B3_B4']},
  {name: 'S4_Base_logratio', bands: ['B2','B3','B4','B8','lr_ln_B2_B3','lr_ln_B2_B4']},
  {name: 'S5_All', bands: [
    'B2','B3','B4','B8',
    'ln_B2','ln_B3','ln_B4','ln_B8',
    'r_B2_B3','r_B2_B4','r_B3_B4',
    'lr_ln_B2_B3','lr_ln_B2_B4'
  ]}
];

// =====================
// FUNGSI
// =====================
function runQuarter(fs) {
    var predictorsFS = featureBank.select(fs.bands);
    // --- stack & sample
    var stack = predictorsFS.addBands(depth);

    var sample = stack.sample({
        region: buffer,
        scale: numScale,
        numPixels: numPixel,
        seed: 42,
        geometries: false
    });

    // =====================
    // TRAIN / TEST SPLIT
    // =====================
    var split = 0.7; // 70% train, 30% test

    // tambahkan kolom random 0..1
    var sampleRand = sample.randomColumn('rand', 42);

    var trainSet = sampleRand.filter(ee.Filter.lt('rand', split));
    var testSet  = sampleRand.filter(ee.Filter.gte('rand', split));

    print('N sample total:', sample.size());
    print('N train:', trainSet.size());
    print('N test:', testSet.size());

    // --- train RF regression (TRAIN SET)
    var rf = ee.Classifier.smileRandomForest(rfParams).setOutputMode('REGRESSION');

    var trainedRF = rf.train({
        features: trainSet,
        classProperty: 'depth',
        inputProperties: ['B2','B3','B4','B8']
    });

    // --- predict raster
    var depth_est = predictorsFS
        .classify(trainedRF)
        .rename('Depth_RF')
        .updateMask(waterMask);

    Map.addLayer(
        depth_est,
        {min: 0, max: 30, palette: ['#E0F3FC','#9BC8DB','#569DB9','#084C67']},
        'Depth RF ' + label + ' (' + startDate + ' to ' + endDate + ')',
        false
    );

    // =====================
    // EVALUASI DI TEST SET
    // =====================
    var testPred = testSet.classify(trainedRF); // kolom prediksi namanya 'classification'

    // R² (Pearson^2) pada TEST
    var corr = testPred.reduceColumns({
        reducer: ee.Reducer.pearsonsCorrelation(),
        selectors: ['depth', 'classification']
    });

    var r = ee.Number(corr.get('correlation'));
    var r2 = r.pow(2);

    // MAE & RMSE pada TEST
    var testMetrics = testPred.map(function(f) {
        var a = ee.Number(f.get('depth'));
        var p = ee.Number(f.get('classification'));
        return f.set({
        absError: a.subtract(p).abs(),
        sqError: a.subtract(p).pow(2)
        });
    });

    // var mae = ee.Number(testMetrics.aggregate_mean('absError'));
    // var rmse = ee.Number(testMetrics.aggregate_mean('sqError')).sqrt();

    print('==============================');
    print('Periode:', label, startDate, 'to', endDate);
    print('TEST R²:', r2);
    // print('TEST MAE:', mae);
    // print('TEST RMSE:', rmse);

    // Chart Actual vs Predicted (TEST)
    var chart = ui.Chart.feature.byFeature(
        testPred,
        'depth',
        ['classification']
    )
    .setChartType('ScatterChart')
    .setOptions({
        title: 'Actual vs Predicted Depth (RF) TEST ' + label + ' ' + startDate + ' to ' + endDate,
        hAxis: {title: 'Actual Depth (GEBCO)'},
        vAxis: {title: 'Predicted Depth (RF)'},
        pointSize: 3,
        trendlines: {0: {showR2: true, visibleInLegend: true}}
    });

    print(chart);
    return depth_est;
}

// =====================
featureSets.forEach(runQuarter);