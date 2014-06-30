package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

import org.apache.spark.mllib.clustering.topicmodeling.documents.Enumerator
import org.apache.spark.mllib.util.LocalSparkContext
import org.scalatest.FunSuite

/**
 * Created by valerij on 6/27/14.
 */
trait AbstractTopicModelSuite[DocumentParameterType <: DocumentParameters,
GlobalParameterType <: GlobalParameters] extends FunSuite with LocalSparkContext {

  val EPS = 1e-5

  def testPLSA(plsa: TopicModel[DocumentParameterType,GlobalParameterType]) {
    val rawDocuments = sc.parallelize(Seq("a b a", "x y y z", "a b z x ").map(_.split(" ").toSeq))

    val docs = Enumerator.numerate(rawDocuments, 0)

    val (theta, global) = plsa.infer(docs)
    val phi = global.phi

    for (topic <- phi) assert(doesSumEqualToOne(topic), "phi matrix is not normalized")

    assert(phi.forall(_.forall(_ >= 0f)), "phi matrix is non-non-negative")

    for (documentParameter <- theta.collect)
      assert(doesSumEqualToOne(documentParameter.theta), "theta is not normalized")

    assert(theta.collect.forall(_.theta.forall(_ >= 0f)), "theta is not non-non-negative")

  }

  private def doesSumEqualToOne(arr: Array[Float]) = math.abs(arr.sum - 1) < EPS

}
