package deep.learning.networks

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid

import scala.util.Random

/**
 * Created by inakov on 4/30/15.
 */
class Network(networkDefinition: DenseVector[Int]) {

  val numberOfLayers: Int = networkDefinition.length
  val definition: DenseVector[Int] = networkDefinition

  val biases: DenseVector[DenseVector[Double]] = {
    for(y <- definition(1 to -1))
      yield DenseVector.rand[Double](y)
  }

  val weights: DenseVector[DenseMatrix[Double]] = {
    val values = for((x, y) <- definition(0 to -2).valuesIterator.zip(definition(1 to -1).valuesIterator).toArray)
      yield DenseMatrix.rand[Double](x, y)
    DenseVector[DenseMatrix[Double]](values)
  }

  println(biases)
  println(weights)

  def feedforward(a: DenseVector[Double]) ={
    var result = a;
    for((b, w) <- biases.valuesIterator.zip(weights.valuesIterator))
      result = sigmoid((w * result) + b)
    result
  }

  def SGD(trainingData: List, epochs: Int, miniBatchSize: Int, eta: Double, testData:List = None): Unit ={
    val trainingDataLength: Int = trainingData.length
    for(j <- 0 to epochs){
      Random.shuffle(trainingData)
      val miniBatches: List[List] =
        for(k <- (0 to trainingDataLength by miniBatchSize).toList) yield trainingData.slice(k, k + miniBatchSize)

      for(miniBatch <- miniBatches)
        updateMiniBatch(miniBatch, eta)

      if (testData != None)
        print("Epoch " + j + ": " + evaluate + " / " + testData.length)
      else
        print("Epoch "+ j + " complete")
    }
  }

  def updateMiniBatch(miniBatch: List, eta: Double): Unit = ???

  def backprop = ???

  def evaluate = ???

  def costDerivative = ???

  def sigmoid_prime = ???

}
