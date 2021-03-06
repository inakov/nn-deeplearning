package deep.learning.networks

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats.distributions.Rand

import scala.collection.mutable.MutableList
import scala.util.Random

/**
 * Created by inakov on 4/30/15.
 */
class Network(networkDefinition: List[Int]) {

  val numberOfLayers: Int = networkDefinition.length
  val definition: List[Int] = networkDefinition

  var biases: MutableList[DenseVector[Double]] = MutableList[DenseVector[Double]]()
  var weights: MutableList[DenseMatrix[Double]] = MutableList[DenseMatrix[Double]]()

  for(y <- definition.drop(1).take(definition.length-1)) biases += DenseVector.rand[Double](y, rand = breeze.stats.distributions.Gaussian(0,1))

  for((x, y) <- definition.take(definition.length-1).zip(definition.drop(1).take(definition.length-1))) weights += DenseMatrix.rand[Double](x, y, rand = breeze.stats.distributions.Gaussian(0,1))


  def feedforward(a: DenseVector[Double]) ={
    var result = a;

    for((b, w) <- biases.zip(weights))
      result = sigmoid((w.t * result) + b)
    result
  }

  def SGD(trainingData: Seq[(DenseVector[Double],DenseVector[Double])], epochs: Int, miniBatchSize: Int, eta: Double, testData:Seq[(DenseVector[Double],DenseVector[Double])] = Nil): Unit ={
    val trainingDataLength: Int = trainingData.length
    for(j <- 0 until epochs){
      Random.shuffle(trainingData)
      val miniBatches: List[Seq[(DenseVector[Double],DenseVector[Double])]] =
        for(k <- (0 until trainingDataLength by miniBatchSize).toList) yield trainingData.slice(k, k + miniBatchSize)

      for(miniBatch <- miniBatches)
        updateMiniBatch(miniBatch, eta)

      if (testData != Nil)
        println("Epoch " + j + ": " + evaluate(testData) + "/" + testData.length)
      else
        print("Epoch "+ j + " complete")
    }
  }

  def updateMiniBatch(miniBatch: Seq[(DenseVector[Double],DenseVector[Double])], eta: Double): Unit = {
    var nabla_b: MutableList[DenseVector[Double]] = for(b <- biases) yield DenseVector.zeros[Double](b.length)
    var nabla_w: MutableList[DenseMatrix[Double]] = for(w <- weights) yield DenseMatrix.zeros[Double](w.rows, w.cols)

    var delta_nabla_b: MutableList[DenseVector[Double]] = MutableList[DenseVector[Double]]()
    var delta_nabla_w: MutableList[DenseMatrix[Double]] = MutableList[DenseMatrix[Double]]()

    for((x, y) <- miniBatch){
      val backpropResult = backprop(x, y)
      delta_nabla_b = backpropResult._1
      delta_nabla_w = backpropResult._2

      nabla_b = for((nb, dnb) <- nabla_b.zip(delta_nabla_b)) yield nb+dnb
      nabla_w = for((nw, dnw) <- nabla_w.zip(delta_nabla_w)) yield nw+dnw
    }
    weights = for((w, nw) <- weights.zip(nabla_w)) yield w-(eta/miniBatch.length)*nw
    biases = for((b, nb) <- biases.zip(nabla_b)) yield b-(eta/miniBatch.length)*nb
  }


  def backprop(x: DenseVector[Double], y: DenseVector[Double]):(MutableList[DenseVector[Double]], MutableList[DenseMatrix[Double]]) ={
    var nabla_b: MutableList[DenseVector[Double]] = for(b <- biases) yield DenseVector.zeros[Double](b.length)
    var nabla_w: MutableList[DenseMatrix[Double]] = for(w <- weights) yield DenseMatrix.zeros[Double](w.rows, w.cols)

    var activation: DenseVector[Double] = x
    var activations: MutableList[DenseVector[Double]] = MutableList[DenseVector[Double]](x)
    var zs: MutableList[DenseVector[Double]] = MutableList[DenseVector[Double]]()

    //feedforward
    for((b, w) <- biases.zip(weights)){
      val z = (w.t * activation) + b
      zs += z
      activation = sigmoid(z)
      activations += activation
    }
    //backward pass
    var delta = costDerivative(activations.reverse.head, y) :* sigmoidPrime(zs.reverse.head)
    nabla_b(nabla_b.length-1) = delta
    nabla_w(nabla_w.length-1) = delta * activations(activations.length-2).t
    for(l <- 2 until numberOfLayers){
      val z = zs(zs.length-l)
      val spv = sigmoidPrime(z);
      delta = (weights(weights.length-1) * delta) :* spv
      nabla_b(nabla_b.length-l) = delta
      nabla_w(nabla_w.length-l) = delta * activations(activations.length-l-1).t
    }

    (nabla_b, nabla_w)
  }

  def evaluate(testData: Seq[(DenseVector[Double],DenseVector[Double])]) = {

    val testResult:Seq[(Int, DenseVector[Double])] = for((x,y) <- testData) yield (argmax(feedforward(x)), y)
    var result = 0;
    for((x,y) <- testResult; if y(x) == 1.0){
      result+=1;
    }
    result
  }

  def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
    outputActivations-y
  }

  def sigmoidPrime(z: DenseVector[Double]): DenseVector[Double]= {
    (sigmoid(z):*(1.0-sigmoid(z)))
  }

}
