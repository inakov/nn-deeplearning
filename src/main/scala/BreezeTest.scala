/**
 * Created by inakov on 4/30/15.
 */
import breeze.linalg._
import breeze.numerics.sigmoid
import deep.learning.networks.Network
import training.data.loader.MnistLoader

//import deep.learning.networks.Network

//http://neuralnetworksanddeeplearning.com/chap1.html
object BreezeTest {

  def main(args: Array[String]) {
//    val net = new Network(List(784, 30, 10))
    val v1 = DenseVector(1,2,3)
    val v2 = DenseVector(4,5,6)

    println(1-v1)
//    val mnistDataLoader = new MnistLoader
//    val mnistDataset = mnistDataLoader.loadTrainingDataSet()
//
//    val trainingData = mnistDataset._1
//    val validationData = mnistDataset._2
//
//    println(trainingData.length)
//    println(validationData.length)
  }

}
