import java.nio.file.{Files, Paths}

import scala.annotation.tailrec
import scala.collection.JavaConversions._
import scala.collection.{LinearSeq, mutable}

object GeneratePlot {

  def main(args: Array[String]): Unit = {
    val aucs = mutable.MutableList[Double]()
    for (predictionNumber <- 2 to 30) {
      val fileLines = readFileLines(predictionNumber)
      val labels2Probabilities = fileLines.map(getLabel2Probability)
      val (fpr2Tpr, auc) = calcFprTprAuc(labels2Probabilities)
      writePlotData(predictionNumber, fpr2Tpr)
      aucs += auc
    }

    writeAucs2File(aucs)
  }

  def getLabel2Probability: (String) => (Int, Double) = {
    string => {
      val dissected = string.split(',')
      val label = dissected(0).toInt
      val probability = dissected(1).toDouble
      (label, probability)
    }
  }

  def readFileLines(predictionNumber: Int): List[String] = {
    Files.readAllLines(Paths.get("res", "result", s"result_$predictionNumber.csv")).drop(1).toList
  }

  def calcFprTprAuc(labels2Probabilities: List[(Int, Double)]): (List[(Double, Double)], Double) = {
    val posCount = labels2Probabilities.count(p => p._1 == 1).toDouble
    val negCount = labels2Probabilities.count(p => p._1 == 0).toDouble

    @tailrec
    def iter(xs: List[(Int, Double)],
             fpr2tpr: List[(Double, Double)] = (0.0, 0.0) :: Nil,
             auc: Double = 0): (List[(Double, Double)], Double) = {
      if (xs.isEmpty) {
        (fpr2tpr, auc)
      } else {
        val label = xs.head._1
        val (fprLast, tprLast) = fpr2tpr.last
        if (label == 0) {
          val (newFpr, newTpr) = (fprLast + 1.0 / negCount, tprLast)
          iter(xs.tail, fpr2tpr :+ (newFpr, newTpr), auc + 1.0 / negCount * tprLast)
        } else {
          val (newFpr, newTpr) = (fprLast, tprLast + 1.0 / posCount)
          iter(xs.tail, fpr2tpr :+ (newFpr, newTpr), auc)
        }
      }
    }

    iter(labels2Probabilities)
  }

  def writePlotData(predictionNumber: Int, fpr2Tpr: List[(Double, Double)]): Unit = {
    val plotText = "FPR,TPR" +: (for (p <- fpr2Tpr) yield s"${p._1},${p._2}")
    Files.write(Paths.get("res", "plot", s"plot_$predictionNumber.csv"), plotText)
  }

  def writeAucs2File(aucs: LinearSeq[Double]): Any = {
    Files.write(Paths.get("res", "auc", "by-diy", "all-aucs"), aucs.zipWithIndex.map {
      case (auc, i) => s"$i -> $auc"
    })
  }
}

