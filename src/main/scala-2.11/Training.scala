import java.nio.file.{Files, Paths}

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.JavaConversions._
import scala.collection.{LinearSeq, mutable}

object Training {
  def main(args: Array[String]): Unit = {
    val sparkSession = createSparkSession

    val data = readData(sparkSession, "cs-training")

    data.cache()
    data.createOrReplaceTempView("data")

    val (training, testing) = getTrainingAndTesting(sparkSession, data)

    val evaluator = new BinaryClassificationEvaluator

    val aucs = mutable.MutableList[Double]()

    for (maxDepth <- 2 to 30) {
      val decisionTreeClassifier = createDTClassifier(maxDepth)
      val model = decisionTreeClassifier.fit(training)
      val predictions = model.transform(testing)

      val auc = evaluator.evaluate(predictions)

      aucs += auc

      val labels2Probabilities = getLabels2Probabilities(predictions)
      val csv = convert2Csv(labels2Probabilities)

      write(csv, s"result_$maxDepth.csv")
    }

    writeAucs2File(aucs)
  }

  def write(csv: Array[String], fileName: String) = {
    Files.write(Paths.get("res", "result", fileName), csv.toList)
  }

  def convert2Csv(labels2Probabilities: RDD[(Int, Double)]): Array[String] = {
    "Label,Probability" +: labels2Probabilities.collect().map(p => s"${p._1},${p._2}")
  }

  def getLabels2Probabilities(predictions: DataFrame): RDD[(Int, Double)] = {
    for (row <- predictions.rdd)
      yield (row.getAs[Double]("label").toInt, row.getAs[DenseVector]("probability")(1))
  }

  def createDTClassifier(maxDepth: Int): DecisionTreeClassifier = {
    new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(maxDepth)
      .setMaxBins(32)
  }

  def readData(sparkSession: SparkSession, fileName: String): DataFrame = {
    import sparkSession.sqlContext.implicits._
    sparkSession.sqlContext.read.csv(fileName)
      .rdd.map(extractFeatures).toDF("index", "label", "features")
  }

  def extractFeatures(row: Row): (Int, Double, SparseVector) = {
    val index = row.getAs[String](0).toInt
    val labelString: String = row.getAs[String](1)
    val label = if (labelString.isEmpty) 0 else labelString.toDouble
    val features = for (i <- 2 to 11) yield convert2Double(row.getAs[String](i))
    (index, label, new SparseVector(10, (0 until 10).toArray, features.toArray))
  }

  def convert2Double(input: String): Double = {
    if (input.equalsIgnoreCase("NA"))
      0.0
    else
      input.toDouble
  }

  def createSparkSession: SparkSession = {
    SparkSession.builder().appName("digits-recognition").master("local[4]").getOrCreate()
  }

  def getTrainingAndTesting(sparkSession: SparkSession, data: DataFrame): (DataFrame, DataFrame) = {
    val totalCount = data.count()
    val threshold = totalCount.toDouble * 0.7
    val training = sparkSession.sql(s"SELECT * from data WHERE index <= $threshold")
    val testing = sparkSession.sql(s"SELECT * from data WHERE index > $threshold")

    training.cache()
    testing.cache()

    (training, testing)
  }

  def writeAucs2File(aucs: LinearSeq[Double]): Any = {
    Files.write(Paths.get("res", "auc", "by-spark", "all-aucs"), aucs.zipWithIndex.map {
      case (auc, i) => s"$i -> $auc"
    })
  }
}
