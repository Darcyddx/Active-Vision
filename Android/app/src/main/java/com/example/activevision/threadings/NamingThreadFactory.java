package com.example.activevision.threadings;

import java.util.concurrent.ThreadFactory;

public class NamingThreadFactory implements ThreadFactory {
    private final String namePrefix;
    private int counter = 0;

    public NamingThreadFactory(String namePrefix) {
        this.namePrefix = namePrefix;
    }

    @Override
    public Thread newThread(Runnable runnable) {
        Thread thread = new Thread(runnable);
        thread.setName(namePrefix + "-" + counter++);
        return thread;
    }
}
