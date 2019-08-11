/*
 *    OzaBagBMod1.java
 *    Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
 *    @author Bernhard Pfahringr (bernhard@waikato.ac.nz)
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


package moa.classifiers.meta;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Multithreading;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.FlagOption;

import java.util.Random;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classifier to train</li>
 * <li>-n : The ensemble size</li>
 * <li>-p : Run in parallel</li>
 * <li>-s : The random seed</li> </ul>
 *
 * @author Bernhard Pfahringer (bernhard@waikato.ac.nz)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class OzaBagB extends AbstractClassifier implements MultiClassClassifier, Multithreading {

    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 2L;

    public ClassOption _baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption _ensembleSizeOption = new IntOption("ensembleSize", 'n',
            "The ensemble size.", 10, 1, Integer.MAX_VALUE);

    public FlagOption _parallelOption = new FlagOption("parallel", 'p',
            "Run ensemble in parallel.");

    public IntOption _randomSeedOption = new IntOption("randomSeed", 's',
            "The random seed.", 42, -Integer.MAX_VALUE, Integer.MAX_VALUE);

    public IntOption _coreAmountOption = new IntOption("NumberOfCores", 'z',
            "The amount of cores to use for parallelism, Note: The max accounts for physical and virtual Cores", 1, 0, Runtime.getRuntime().availableProcessors());

    protected Classifier[] _classifiers;
    protected Instance _instance;
    protected Random _r;
    protected int[] _weight;
    protected ForkJoinPool _threadpool;
    protected double _cpuTime;
    protected double _t1;

    public int getCores(){
        return _coreAmountOption.getValue();
    }

    public void resetLearningImpl() {
        _r = new Random( _randomSeedOption.getValue());
        int ensembleSize = _ensembleSizeOption.getValue();
//        int cores = _coreAmountOption.getValue();
//        if(cores == 0){
//            _threadpool = new ForkJoinPool(_coreAmountOption.getMaxValue());
//        }else
//        _threadpool = new ForkJoinPool(cores); // how many worker threads to use e.g how many cores to use

        Classifier baseLearner = (Classifier) getPreparedClassOption(_baseLearnerOption);
        baseLearner.resetLearning();
        _classifiers = new Classifier[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) {
            _classifiers[i] = (Classifier) baseLearner.copy();
        }
        _weight = new int[ensembleSize];
    }


    public void trainOnInstanceImpl(Instance inst) throws ExecutionException, InterruptedException {

        double t1 = System.currentTimeMillis();
        _t1 = t1;



        int n = _classifiers.length;
        for (int i = 0; i < n; i++) _weight[i] = MiscUtils.poisson(1.0, _r);

        if (_parallelOption.isSet()) {

            _threadpool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> train(i, inst))).get();

        } else {
            for (int i = 0; i < n; i++) train(i, inst);
        }
    }

    //send the overall CPU time of the parallel model to the Evaluator to update the stats in the GUI
    public double getCpuTime(){
        return _cpuTime;
    }

    public void train(int index, Instance instance) {

        int k = _weight[index];
        if (k > 0) {
            Instance weightedInst = (Instance) instance.copy();
            weightedInst.setWeight(instance.weight() * k);
            _classifiers[index].trainOnInstance(weightedInst);
        }
        double t2 = System.currentTimeMillis();
        _cpuTime += (t2 - _t1);
    }


    public double[] getVotesForInstance(Instance instance) {
        _t1 = System.currentTimeMillis();

        if (_parallelOption.isSet()) {
            double sum = 0.0;
            _instance = instance;
            double[] votes =
                    Arrays.asList(_classifiers)
                    .parallelStream()
                    .collect(Predictor::new, Predictor::accept, Predictor::combine)
                    .getVotes();

            for (double v: votes) sum += v;
       //    sum = Arrays.stream(votes).parallel().reduce(0,Double::sum);
            if (sum > 0.0) {
                sum = 1.0/sum;
                final double sumF = sum;
//                votes = Arrays.stream(votes).parallel().map(i -> i* sumF).toArray();
                for (int i = 0; i < votes.length; i++) {
                    votes[i] *= sum;
                }

            }
            double t2 = System.currentTimeMillis();
            _cpuTime += (t2 - _t1);
            return votes;

        } else {

            DoubleVector combinedVote = new DoubleVector();
            for (Classifier cl: _classifiers) {
                DoubleVector vote = new DoubleVector(cl.getVotesForInstance(instance));
                combinedVote.addValues(vote);
            }
            combinedVote.normalize();
            double t2 = System.currentTimeMillis();
            _cpuTime += (t2 - _t1);
            return combinedVote.getArrayRef();
        }

    }

    // Avoids Thread Pool Leaking
    public void trainingHasEnded(){
        _threadpool.shutdown();

    }

    public boolean isRandomizable() {
        return true;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size", _classifiers == null ? 0 : _classifiers.length)};
    }

    public Classifier[] getSubClassifiers() {
        return Arrays.copyOf(_classifiers, _classifiers.length);
    }

    public void ReceivePool(ForkJoinPool pool){
        System.out.println("Start time");
        _threadpool = pool;
    }


    //======================================================================
    //
    // utility class for the parallel "collect" of predictions
    //
    //======================================================================

    class Predictor
    {
        private double[] _votes;

        public double[] getVotes() { return _votes; }

        public void updateVotes(double[] a, double[] b) {

            double[] target = (a.length >= b.length) ? a : b;
            double[] source = (a.length >= b.length) ? b : a;

            for (int i = 0; i < source.length; i++) {
                target[i] += source[i];
            }
            _votes = target;
        }

        public void accept(Classifier cl) {
            double[] votes = cl.getVotesForInstance(_instance);
            if (_votes == null) {
                _votes = votes;
            } else {
                updateVotes(_votes, votes);
            }
        }

        public void combine(Predictor other) {
            updateVotes(_votes, other._votes);
        }

    }


}