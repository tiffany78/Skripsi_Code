var buffer = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Buffer24Mil_Asmat");
Map.centerObject(buffer, 8);

// PARAMETER
var csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var gebco2025 = ee.Image("projects/ee-tiffanytasyaagatha/assets/gebco2025");

var numScale = 20;
var numPixel = 250;
var rfParams = {
  numberOfTrees: 50,       // saya naikkan sedikit biar stabil
  variablesPerSplit: null,
  minLeafPopulation: 5,
  bagFraction: 0.7,
  seed: 42
};

function runYearWithFeatureLoop(startDate, endDate, label) {

  // 1) composite median 1 tahun
  var composite = s2
    .filterDate(startDate, endDate)
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    })
    .median()
    .clip(buffer);

  // 2) water mask
  var ndwi = composite.normalizedDifference(['B3','B8']);
  var waterMask = ndwi.gt(0);

  var img_water = composite.updateMask(waterMask);

  // 3) predictors dasar (reflectance 0-1)
  var predictorsBase = img_water
    .select(['B2','B3','B4','B8'])
    .multiply(0.0001)
    .toFloat();

  // 4) depth training (GEBCO)
  var depth = gebco2025
    .updateMask(gebco2025.lt(0))
    .multiply(-1)
    .rename('depth')
    .resample('bilinear')
    .updateMask(waterMask)
    .toFloat();

  depth = depth.reproject({
    crs: predictorsBase.select('B2').projection(),
    scale: 20
  });

  // ====== 5) BUAT FITUR TAMBAHAN SEKALI SAJA ======
  var eps = 1e-6;

  var lnBands = predictorsBase.add(eps).log()
    .rename(['ln_B2','ln_B3','ln_B4','ln_B8']);

  var r_B2B3 = predictorsBase.select('B2')
    .divide(predictorsBase.select('B3').add(eps)).rename('r_B2_B3');
  var r_B2B4 = predictorsBase.select('B2')
    .divide(predictorsBase.select('B4').add(eps)).rename('r_B2_B4');
  var r_B3B4 = predictorsBase.select('B3')
    .divide(predictorsBase.select('B4').add(eps)).rename('r_B3_B4');

  var B2 = predictorsBase.select('B2').add(eps);
  var B3 = predictorsBase.select('B3').add(eps);
  var B4 = predictorsBase.select('B4').add(eps);

  var lr_B2B3 = B2.log().divide(B3.log()).rename('lr_ln_B2_B3');
  var lr_B2B4 = B2.log().divide(B4.log()).rename('lr_ln_B2_B4');

  // gabungkan semua fitur jadi satu "bank fitur"
  var featureBank = predictorsBase
    .addBands(lnBands)
    .addBands([r_B2B3, r_B2B4, r_B3B4, lr_B2B3, lr_B2B4])
    .updateMask(waterMask);

  // ====== 6) DAFTAR SET FITUR YANG MAU DITES ======
  // Kamu tinggal tambah/kurangi kombinasi di sini.
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

  // ====== 7) FUNGSI EVALUASI PER SET FITUR ======
  function evalOneSet(fs) {

    var predictors = featureBank.select(fs.bands);

    // stack & sample
    var stack = predictors.addBands(depth);

    var sample = stack.sample({
      region: buffer,
      scale: numScale,
      numPixels: numPixel,
      seed: 42,
      geometries: false
    });

    // train/test split
    var split = 0.7;
    var sampleRand = sample.randomColumn('rand', 42);
    var trainSet = sampleRand.filter(ee.Filter.lt('rand', split));
    var testSet  = sampleRand.filter(ee.Filter.gte('rand', split));

    // train RF
    var rf = ee.Classifier.smileRandomForest(rfParams).setOutputMode('REGRESSION');

    var trainedRF = rf.train({
      features: trainSet,
      classProperty: 'depth',
      inputProperties: fs.bands
    });

    // evaluasi di test
    var testPred = testSet
        .classify(trainedRF);

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

    // <=30m metrics
    var testShallow = testPred.filter(ee.Filter.lte('depth', 30));
    var corr30 = testShallow.reduceColumns({
      reducer: ee.Reducer.pearsonsCorrelation(),
      selectors: ['depth', 'classification']
    });

    var r30 = ee.Number(corr30.get('correlation'));
    var r2_30 = r30.pow(2);

    var met30 = testShallow.map(function(f){
      var a = ee.Number(f.get('depth'));
      var p = ee.Number(f.get('classification'));
      return f.set({
        absError: a.subtract(p).abs(),
        sqError: a.subtract(p).pow(2)
      });
    });

    var mae30 = ee.Number(met30.aggregate_mean('absError'));
    var rmse30 = ee.Number(met30.aggregate_mean('sqError')).sqrt();

    print('==============================');
    print('Year:', label, startDate, 'to', endDate);
    print('FeatureSet:', fs.name, 'Bands:', fs.bands);
    print('TEST R²:', r2);
    // pritn('MAE:', mae, 'RMSE:', rmse);
    // print('TEST (<=30m) R²:', r2_30, 'MAE:', mae30, 'RMSE:', rmse30);

    // (opsional) kalau kamu mau lihat peta hanya untuk set tertentu
    // var depth_est = predictors.classify(trainedRF).rename('Depth_RF_' + fs.name).updateMask(waterMask);
    // Map.addLayer(depth_est, {min:0, max:40, palette:['#E0F3FC','#9BC8DB','#569DB9','#084C67']},
    //   'Depth RF ' + fs.name, false);
  }

  // ====== 8) JALANKAN LOOP FITUR ======
  featureSets.forEach(evalOneSet);
}

// Jalankan hanya median 2025
runYearWithFeatureLoop("2025-01-01", "2026-01-01", "Median2025");