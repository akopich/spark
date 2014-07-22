/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering.topicmodeling.topicmodels

import org.scalatest.FunSuite

import org.apache.spark.mllib.clustering.topicmodeling.documents.Enumerator
import org.apache.spark.mllib.util.LocalSparkContext

trait AbstractTopicModelSuite[DocumentParameterType <: DocumentParameters,
    GlobalParameterType <: GlobalParameters] extends FunSuite with LocalSparkContext {

  val EPS = 1e-5

  def testPLSA(plsa: TopicModel[DocumentParameterType,GlobalParameterType]) {
    // the data in text form
    val rawDocuments = sc.parallelize(Seq("a b a", "x y y z", "a b z x ").map(_.split(" ").toSeq))

    // use enumerator to map documents in vector space
    val docs = Enumerator.numerate(rawDocuments, 0)

    // training plsa
    val (docParameters, global) = plsa.infer(docs)

    // this matrix is an array of topic distributions over words
    val phi = global.phi

    // thus, every its row should sum up to one
    for (topic <- phi) assert(doesSumEqualToOne(topic), "phi matrix is not normalized")

    assert(phi.forall(_.forall(_ >= 0f)), "phi matrix is non-non-negative")

    // a distribution of a document over topics (theta) should also sum up to one
    for (documentParameter <- docParameters.collect)
      assert(doesSumEqualToOne(documentParameter.theta), "theta is not normalized")

    assert(docParameters.collect.forall(_.theta.forall(_ >= 0f)), "theta is not non-non-negative")

  }

  private def doesSumEqualToOne(arr: Array[Float]) = math.abs(arr.sum - 1) < EPS

}
