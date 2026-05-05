METODE 1
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).multiply(0.0001);
}
var dataset = s2
              .filterDate('2024-01-01', '2025-01-01')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
              .map(maskS2clouds)
              .filterBounds(area2);
var composite = dataset.median().clip(area2);

METODE 2
var QA_BAND = 'cs_cdf';
var CLEAR_THRESHOLD = 0.55;

var start = ee.Date.fromYMD(year, 1, 1);
var end = start.advance(1, 'year');
var collection = s2
.filterDate(start, end)
.filterBounds(area2)
.linkCollection(csPlus, [QA_BAND])
.map(function(img) {
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
});
var compositeMedian = collection.median();