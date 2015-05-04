package training.data.loader

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.JavaConverters._

/**
 * Created by inakov on 5/4/15.
 */
class MnistLoader {

  val IMAGES_FILE_PATH: String = "/home/inakov/IdeaProjects/branches/branches/branches/nn-deeplearning/src/main/resources/train-images-idx3-ubyte"
  val LABELS_FILE_PATH: String = "/home/inakov/IdeaProjects/branches/branches/branches/nn-deeplearning/src/main/resources/train-labels-idx1-ubyte"

  def loadTrainingDataSet():(Seq[(DenseVector[Double],DenseVector[Double])], Seq[(DenseVector[Double],DenseVector[Double])]) = {
    val dataSet = MNISTDataUtil.loadTrainingData(IMAGES_FILE_PATH, LABELS_FILE_PATH)
    val imagesList = dataSet.getFirst.asScala
    val labelList = dataSet.getSecond.asScala

    val trainingInputs: Seq[DenseVector[Double]] = for(image <- imagesList) yield DenseVector(image.flatten.map(_.toDouble)).toDenseVector
    val trainingResult: Seq[DenseVector[Double]] = for(label <- labelList) yield vectorizedLabel(label)

    val trainingData = trainingInputs.zip(trainingResult)

    (trainingData.take(50000), trainingData.drop(50000))
  }

  def vectorizedLabel(label: Int): DenseVector[Double] = {
    val result = DenseVector.zeros[Double](10)
    result(label) = 1.0

    result
  }
}