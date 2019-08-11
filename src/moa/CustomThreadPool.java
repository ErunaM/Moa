package moa;

import java.util.concurrent.ForkJoinPool;

public class CustomThreadPool {
    int[] threadIDs;
    ForkJoinPool pool;

    public CustomThreadPool(int size){
        pool = new ForkJoinPool(size);
        threadIDs = new int[size];

    }
    public int[] getThreadIDs(){


        return null;
    }

}
