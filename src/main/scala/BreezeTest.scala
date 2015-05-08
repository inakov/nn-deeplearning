/**
 * Created by inakov on 4/30/15.
 */
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Rand, Gaussian}
import deep.learning.networks.Network
import training.data.loader.MnistLoader

//import deep.learning.networks.Network

//http://neuralnetworksanddeeplearning.com/chap1.html
object BreezeTest {

  def main(args: Array[String]) {
    val mnistDataLoader = new MnistLoader
    val mnistDataset = mnistDataLoader.loadTrainingDataSet()
    val trainingData = mnistDataset._1

    val validationData = mnistDataset._2

    val net = new Network(List(784, 30, 10))
    net.SGD(trainingData, 10000, 10, 1.0, testData = validationData)
//    println(DenseMatrix.rand[Double](10, 3, rand = Rand.gaussian))
  }

}
