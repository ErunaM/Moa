package moa.classifiers;

import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

public interface Multithreading {

     void trainingHasEnded();

     AtomicInteger getCpuTime();

     void init() throws InterruptedException, ExecutionException;


}
