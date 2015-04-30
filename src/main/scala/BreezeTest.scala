/**
 * Created by inakov on 4/30/15.
 */
import breeze.linalg._
import breeze.numerics.sigmoid
import deep.learning.networks.Network

object BreezeTest {

  def main(args: Array[String]) {
    val net = new Network(DenseVector(2,3,1))
  }

}
