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
    val mnistDataLoader = new MnistLoader
    val mnistDataset = mnistDataLoader.loadTrainingDataSet()
    val trainingData = mnistDataset._1

    val validationData = mnistDataset._2

    val net = new Network(List(784, 30, 10))
    net.SGD(trainingData, 30, 10, 3.0, testData = Nil)
//    val z = DenseVector(8980.542542230683, 10069.722927383342, 9818.197266957031, 9183.099306338094, 9179.353050941767, 8191.41163566096, 8304.119946089011, 8271.75180187576, 8936.320531658464, 8393.302055646922, 8596.419000663935, 8806.758984191983, 8628.02848748314, 8416.915780385174, 8729.813057530013, 9261.794661505506, 8477.36362834955, 8849.378143220963, 9088.935080634586, 9381.416059959554, 9319.563402062871, 9040.443616084514, 8726.46664833916, 9763.85030723229, 8696.709835664422, 8778.637301972225, 10006.319031221841, 8228.428186244031, 8484.350728243428, 9253.329249525748)
//    println("sigmoid(z): " + sigmoid(z))
//    println("1.0-sigmoid(z): " + (1.0-sigmoid(z)))
//    println("prime sigmoid(z): " + (sigmoid(z):*(1.0-sigmoid(z))))
  }

}
