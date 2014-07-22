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

import org.apache.spark.broadcast.Broadcast


/**
 * contains number of words, generated by topic, and words from background.
 * @param wordsFromTopic
 * @param backgroundWords background topics counter
 * @param alphabetSize
 */
private[topicmodels] class RobustGlobalCounters(wordsFromTopic: Array[Array[Float]],
                           val backgroundWords: Array[Float],
                           alphabetSize: Int) extends GlobalCounters(wordsFromTopic, alphabetSize) {

  /**
   * merges two GlobalParameters into a single one
   * @return GlobalParameters
   */
  def + (that: RobustGlobalCounters) = {
    super. + (that)

    for (i <- 0 until alphabetSize) {
      backgroundWords(i) += that.backgroundWords(i)
    }

    new RobustGlobalCounters(wordsFromTopic, backgroundWords, alphabetSize)
  }

  /**
   * adds a local parameter to global parameters
   * @param that DocumentParameters.
   * @param topics broadcasted words by topics distribution
   * @param background words by background distribution
   * @param eps weight of noise
   * @param gamma weight of background
   * @param alphabetSize number of unique words
   * @return GlobalParameters
   */
  def add(that: RobustDocumentParameters,
      topics: Broadcast[Array[Array[Float]]],
      background: Array[Float],
      eps: Float,
      gamma: Float,
      alphabetSize: Int) = {

    val (wordsFromTopicInDoc, wordsFromBackground) =
      that.wordsFromTopicsAndWordsFromBackground(topics, background, eps: Float, gamma)

    wordsFromTopicInDoc.zip(wordsFromTopic).foreach { case (topic, words) =>
      topic.activeIterator.foreach { case (word, num) =>
        words(word) += num
      }
    }

    wordsFromBackground.activeIterator.foreach { case (key, value) =>
      backgroundWords(key) += value
    }

    this
  }
}

/**
 * companion object of class GlobalParameters
 */
private[topicmodels] object RobustGlobalCounters {
  /**
   * construct new GlobalParameters
   * @param topicNum number of topics
   * @param alphabetSize number of unique words
   * @return  new GlobalParameters
   */
  def apply(topicNum: Int, alphabetSize: Int) = {
    val topicWords = Array.ofDim[Float](topicNum, alphabetSize)
    val backgroundWords = new Array[Float](alphabetSize)
    new RobustGlobalCounters(topicWords, backgroundWords, alphabetSize)
  }
}
