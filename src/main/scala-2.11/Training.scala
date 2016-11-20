import java.nio.file.{Files, Paths}

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.JavaConversions._
import scala.collection.{LinearSeq, mutable}

object Training {
  def main(args: Array[String]): Unit = {
    val sparkSession = createSparkSession

    val data = readData(sparkSession, "cs-training.csv")
    data.cache()
    val (training, testing) = getTrainingAndTesting(sparkSession, data)

    val aucs = mutable.MutableList[Double]()
    val precisions = mutable.MutableList[Double]()
    val accuracies = mutable.MutableList[Double]()
    val depthsRange = 6 to 6

    for (maxDepth <- depthsRange) {
      val decisionTreeClassifier = createDTClassifier(maxDepth)
      val model = decisionTreeClassifier.fit(training)
      val predictions = model.transform(testing)
      val scoreAndLabels = getScoreAndLabels(predictions)

      val thresholdBinaries = 100
      val metric = new BinaryClassificationMetrics(scoreAndLabels, thresholdBinaries)

      val roc = getRoc(metric)
      writeRocCsv(maxDepth, roc)

      aucs += getAuc(metric)
      precisions += getPrecision(metric)
      accuracies += getAccuracy(thresholdBinaries, scoreAndLabels)

      val predictionResult = convert2Csv(scoreAndLabels)
      write(predictionResult, s"res/predictions/prediction_$maxDepth.csv")
    }

    val convertToText: LinearSeq[_] => Array[String] = {
      seq => seq.zip(depthsRange).map {
        case (elem, i) => s"$i -> $elem"
      }.toArray
    }

    val aucsAsText = convertToText(aucs)
    val precisionsAsText = convertToText(precisions)
    val accuraciesAsText = convertToText(accuracies)

    write(aucsAsText, "res/metrics/auc/all-aucs")
    write(precisionsAsText, "res/metrics/precision/all-precisions")
    write(accuraciesAsText, "res/metrics/accuracy/all-accuracies")
  }

  def createSparkSession: SparkSession = {
    SparkSession.builder().appName("digits-recognition").master("local[4]").getOrCreate()
  }

  def readData(sparkSession: SparkSession, fileName: String): DataFrame = {
    import sparkSession.sqlContext.implicits._
    sparkSession.sqlContext.read.csv(fileName)
      .rdd.map(extractFeatures).toDF("index", "label", "features")
  }

  def convert2Double(input: String): Double = {
    if (input.equalsIgnoreCase("NA"))
      0.0
    else
      input.toDouble
  }

  def extractFeatures(row: Row): (Int, Double, SparseVector) = {
    val index = row.getAs[String](0).toInt
    val labelString: String = row.getAs[String](1)
    val label = if (labelString.isEmpty) 0 else labelString.toDouble
    val features = for (i <- 2 to 11) yield convert2Double(row.getAs[String](i))
    (index, label, new SparseVector(10, (0 until 10).toArray, features.toArray))
  }

  def getTrainingAndTesting(sparkSession: SparkSession, data: DataFrame): (DataFrame, DataFrame) = {
    val Array(training, testing) = data.randomSplit(Array(0.6, 0.4))

    training.cache()
    testing.cache()

    (training, testing)
  }

  def createDTClassifier(maxDepth: Int): DecisionTreeClassifier = {
    new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(maxDepth)
      .setMaxBins(1024)
  }

  def getScoreAndLabels(predictions: DataFrame): RDD[(Double, Double)] = {
    for (row <- predictions.rdd)
      yield (row.getAs[DenseVector]("probability")(1), row.getAs[Double]("label"))
  }

  def getRoc(metric: BinaryClassificationMetrics): Array[(Double, Double)] = {
    metric.roc().collect()
  }

  def writeRocCsv(maxDepth: Int, roc: Array[(Double, Double)]): Unit = {
    val rocCsv = "FPR,TPR" +: roc.map(p => s"${p._1},${p._2}")
    write(rocCsv, s"res/metrics/roc/roc_$maxDepth.csv")
  }

  def getPrecision(metric: BinaryClassificationMetrics): Double = {
    metric.precisionByThreshold().collect().map(_._2).max
  }

  def getAuc(metric: BinaryClassificationMetrics): Double = {
    metric.areaUnderROC()
  }

  def getAccuracy(thresholdBinaries: Int, scoreAndLabels: RDD[(Double, Double)]): Double = {
    val totalCount = scoreAndLabels.count()
    (0 to thresholdBinaries).map(i => i.toDouble / thresholdBinaries.toDouble).map(i => {
      val rightPredictedCount = scoreAndLabels.map {
        case (score, label) => (math.signum(score - i).toInt + 1, label.toInt)
      }.collect().count(p => p._2 == p._1)

      rightPredictedCount.toDouble / totalCount.toDouble
    }).max
  }

  def convert2Csv(scoreAndLabels: RDD[(Double, Double)]): Array[String] = {
    "Label,Probability" +: scoreAndLabels.collect().map {
      case (score, label) => s"$label,$score"
    }
  }

  def write(csv: Array[String], filePath: String) = {
    val path = filePath.split('/')
    Files.write(Paths.get(path.head, path.tail: _*), csv.toList)
  }
}
