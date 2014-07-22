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

import breeze.linalg.SparseVector

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.topicmodeling.documents.Document
import org.apache.spark.mllib.clustering.topicmodeling.topicmodels.regulaizers.DocumentOverTopicDistributionRegularizer

/**
 * the class contains document parameter in PLSA model
 * @param document
 * @param theta the distribution over topics
 * @param regularizer
 */
class DocumentParameters(val document: Document, val theta: Array[Float],
                         private val regularizer: DocumentOverTopicDistributionRegularizer)
  extends Serializable {
  protected def getZ(topics: Broadcast[Array[Array[Float]]]) = {
    val topicsValue = topics.value
    val numberOfTopics = topicsValue.size

    document.tokens.mapActivePairs { case (word, n) =>
      (0 until numberOfTopics).foldLeft(0f)((sum, topic) => sum + topicsValue(topic)(word) *
        theta(topic))
    }
  }

  private[topicmodels] def wordsFromTopics(topics: Broadcast[Array[Array[Float]]]):
      Array[SparseVector[Float]] = {
    val Z = getZ(topics)

    wordsToTopicCnt(topics, Z)
  }

  private[topicmodels] def wordsToTopicCnt(topics: Broadcast[Array[Array[Float]]],
                                           Z: SparseVector[Float]): Array[SparseVector[Float]] = {
    val array = Array.ofDim[SparseVector[Float]](theta.size)
    forWithIndex(theta)((topicWeight, topicNum) =>
      array(topicNum) = document.tokens.mapActivePairs { case (word,
      num) => num * topics.value(topicNum)(word) * topicWeight / Z(word)
      })
    array
  }

  protected def forWithIndex(array: Array[Float])(operation: (Float, Int) => Unit) {
    var i = 0
    val size = array.size
    while (i < size) {
      operation(array(i), i)
      i += 1
    }
  }

  private[topicmodels] def assignNewTheta(topics: Broadcast[Array[Array[Float]]],
                                          Z: SparseVector[Float]) {
    val newTheta: Array[Float] = {
      val array = Array.ofDim[Float](theta.size)
      forWithIndex(theta)((weight, topicNum) => array(topicNum) = weight * document.tokens
        .activeIterator.foldLeft(0f) { case (sum, (word, wordNum)) => sum + wordNum * topics
        .value(topicNum)(word) / Z(word)
      })
      array
    }
    regularizer.regularize(newTheta, theta)

    val newThetaSum = newTheta.sum

    forWithIndex(newTheta)((wordsNum, topicNum) => theta(topicNum) = wordsNum / newThetaSum)

  }

  private[topicmodels] def getNewTheta(topicsBC: Broadcast[Array[Array[Float]]]) = {
    val Z = getZ(topicsBC)
    assignNewTheta(topicsBC, Z)

    this
  }

  private[topicmodels]  def priorThetaLogProbability = regularizer(theta)

}


object DocumentParameters extends SparseVectorFasterSum {

  def apply(document: Document, numberOfTopics: Int,
            regularizer: DocumentOverTopicDistributionRegularizer) = {
    val theta = getTheta(numberOfTopics)
    new DocumentParameters(document, theta, regularizer)
  }

  private def getTheta(numberOfTopics: Int) = {
    Array.fill[Float](numberOfTopics)(1f / numberOfTopics)
  }
}
