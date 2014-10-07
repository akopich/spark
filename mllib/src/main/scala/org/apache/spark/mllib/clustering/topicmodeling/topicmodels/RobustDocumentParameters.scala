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

import breeze.linalg.{SparseVector, sum}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.topicmodeling.documents.Document
import org.apache.spark.mllib.clustering.topicmodeling.topicmodels.regulaizers.DocumentOverTopicDistributionRegularizer

/**
 * the class contains document parameter in Robust PLSA model
 *
 * @param document
 * @param theta the distribution over topics
 * @param noise noisiness of words
 * @param regularizer
 */
class RobustDocumentParameters(document: Document,
                               theta: Array[Float],
                               val noise: SparseVector[Float],
                               regularizer: DocumentOverTopicDistributionRegularizer)
  extends DocumentParameters(document, theta, regularizer) {

  protected def getZ(topics: Broadcast[Array[Array[Float]]],
                     background: Broadcast[Array[Float]],
                     eps: Float,
                     gamma: Float) = {
    val topicsValue = topics.value
    val backgroundValue = background.value
    val numberOfTopics = topicsValue.size

    val Z = document.tokens.mapActivePairs { case (word, n) =>
      val sum = (0 until numberOfTopics).foldLeft(0f)((sum, topic) =>
        sum + topicsValue(topic)(word) * theta(topic))
      (eps * noise(word) + gamma * backgroundValue(word) + sum) / (1 + eps + gamma)
    }
    Z
  }

  private[topicmodels] def wordsFromTopicsAndWordsFromBackground(
      topics: Broadcast[Array[Array[Float]]],
      background: Broadcast[Array[Float]],
      eps: Float,
      gamma: Float): (Array[SparseVector[Float]], SparseVector[Float]) = {
    val Z = getZ(topics, background, eps, gamma)

    (super.wordsToTopicCnt(topics, Z), wordToBackgroundCnt(background.value, eps, gamma, Z))
  }


  protected def wordToBackgroundCnt(background: Array[Float],
      eps: Float,
      gamma: Float,
      Z: SparseVector[Float]): SparseVector[Float] = {
    document.tokens.mapActivePairs { case (word, num) =>
      num * background(word) * gamma / Z(word)
    }
  }


  protected def getNoise(eps: Float, Z: SparseVector[Float]) = {
    val newWordsFromNoise = document.tokens.mapActivePairs { case (word,num) =>
      eps * noise(word) * num / Z(word)
    }

    val noiseWordsSum = sum(newWordsFromNoise)

    if (noiseWordsSum > 0) {
      newWordsFromNoise.mapActiveValues(_ / noiseWordsSum)
    } else {
      newWordsFromNoise.mapActiveValues(i => 0f)
    }
  }

  /**
   * calculates a new distribution of this document by topic, corresponding to the new topics
   */
  def getNewTheta(topicsBC: Broadcast[Array[Array[Float]]],
      background: Broadcast[Array[Float]],
      eps: Float,
      gamma: Float) = {
    val Z = getZ(topicsBC, background, eps, gamma)

    super.assignNewTheta(topicsBC, Z)

    val newNoise: SparseVector[Float] = getNoise(eps, Z)
    new RobustDocumentParameters(document, theta, newNoise, regularizer)
  }


}

/**
 * companion object of DocumentParameters. Create new DocumentParameters and contain some methods
 */
private[topicmodels] object RobustDocumentParameters extends SparseVectorFasterSum {
  /**
   * create new DocumentParameters
   * @param document
   * @param numberOfTopics
   * @param gamma weight of background
   * @param eps weight of noise
   * @return new DocumentParameters
   */
  def apply(document: Document,
      numberOfTopics: Int,
      gamma: Float,
      eps: Float,
      regularizer: DocumentOverTopicDistributionRegularizer) = {
    val wordsNum = sum(document.tokens)
    val noise = document.tokens.mapActiveValues(word => 1f / wordsNum)

    val documentParameters: DocumentParameters = DocumentParameters(document, numberOfTopics,
      regularizer)
    new RobustDocumentParameters(document, documentParameters.theta, noise, regularizer)
  }

}
