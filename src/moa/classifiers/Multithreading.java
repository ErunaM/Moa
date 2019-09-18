package moa.classifiers;

import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

public interface Multithreading {

     void ReceivePool(ForkJoinPool pool);

     void ReceiveHashSet();

     void trainingHasEnded();

     int getCores();

     HashSet<Integer> getCpuTime();

     void init() throws InterruptedException, ExecutionException;


}
