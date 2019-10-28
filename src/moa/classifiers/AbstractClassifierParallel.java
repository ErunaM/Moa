/*
 *    AbstractClassifier.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */

package moa.classifiers;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import moa.capabilities.CapabilitiesHandler;

import com.github.javacliparser.IntOption;

import com.yahoo.labs.samoa.instances.Instance;

public abstract class AbstractClassifierParallel extends AbstractClassifier
        implements Classifier, CapabilitiesHandler { //Learner<Example<Instance>> {


    public IntOption _amountOfCores = new IntOption("coreSize", 'c',
            "The amount of CPU Cores used for multi-threading", 1, 0, Runtime.getRuntime().availableProcessors());


    /** The amount of CPU cores to be run in parallel */
    public int _numOfCores;
    /** _cpuTime stores the total time the program has been running on the cores */
    protected AtomicInteger _cpuTime;
    /** start time used in threads to measure the start of the training program in parallel */
    protected double _t1;

    protected ForkJoinPool _threadpool;


    /**
     * Creates an classifier and setups the random seed option
     * if the classifier is randomizable.
     */
    public AbstractClassifierParallel() {

        if (isRandomizable()) {
            this.randomSeedOption = new IntOption("randomSeed", 'r',
                    "Seed for random behaviour of the classifier.", 1);
        }
    }

    public abstract void trainOnInstanceImpl(Instance inst);

    public void trainingHasEnded(){ if(_threadpool != null)_threadpool.shutdown(); }

    @Override
    public void resetLearning() {
        _cpuTime = new AtomicInteger();
        _numOfCores = _amountOfCores.getValue();
        if(_numOfCores > 1){
            _threadpool = new ForkJoinPool(_numOfCores);
        }
        this.trainingWeightSeenByModel = 0.0;
        if (isRandomizable()) {
            this.classifierRandom = new Random(this.randomSeed);
        }
        resetLearningImpl();
    }





    public AtomicInteger getCpuTime() {
        return _cpuTime;
    }

}
