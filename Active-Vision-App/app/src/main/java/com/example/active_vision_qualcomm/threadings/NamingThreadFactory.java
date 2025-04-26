package com.example.active_vision_qualcomm.threadings;

import java.util.concurrent.ThreadFactory;

public class NamingThreadFactory implements ThreadFactory {
    private final String namePrefix;
    private int counter = 0;

    public NamingThreadFactory(String namePrefix) {
        this.namePrefix = namePrefix;
    }

    @Override
    public Thread newThread(Runnable r) {
        Thread thread = new Thread(r);
        thread.setName(namePrefix + "-" + counter++);
        return thread;
    }

}
