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
 * contains global counters in PLSA model -- holds n_{tw} (Vorontov's notation) counters and
 * alphabet size
 *
 * @param wordsFromTopics
 * @param alphabetSize
 */
class GlobalCounters(val wordsFromTopics: Array[Array[Float]], val alphabetSize: Int)
  extends Serializable {

  /**
   * merges two GlobalParameters into a single one
   * @param that other GlobalParameters
   * @return GlobalParameters
   */
  private[topicmodels] def + (that: GlobalCounters) = {
    wordsFromTopics.zip(that.wordsFromTopics).foreach { case (thisOne, otherOne) =>
      (0 until alphabetSize).foreach(i => thisOne(i) += otherOne(i))
    }

    new GlobalCounters(wordsFromTopics, alphabetSize)
  }

  /**
   * calculates and add local parameters to global parameters
   * @param that DocumentParameters.
   * @param topics broadcasted words by topics distribution
   * @param alphabetSize number of unique words
   * @return GlobalParameters
   */
  private[topicmodels] def add(that: DocumentParameters,
                               topics: Broadcast[Array[Array[Float]]],
                               alphabetSize: Int) = {

    val wordsFromTopic = that.wordsFromTopics(topics)

    wordsFromTopic.zip(wordsFromTopics).foreach { case (topic, words) =>
      topic.activeIterator.foreach{ case (word, num) => words(word) += num }
    }
    this
  }
}

private[topicmodels] object GlobalCounters {
  def apply(topicNum: Int, alphabetSize: Int) = {
    val topicWords = (0 until topicNum).map(i => new Array[Float](alphabetSize)).toArray
    new GlobalCounters(topicWords, alphabetSize)
  }
}
