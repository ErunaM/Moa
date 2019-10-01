package moa.classifiers;

import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

public interface Multithreading {

     void trainingHasEnded();

     double getCpuTime();

     void init() throws InterruptedException, ExecutionException;


}
