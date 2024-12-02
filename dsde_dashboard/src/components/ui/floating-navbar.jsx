'use client';
import React, { useState } from 'react';
import {
    motion,
    AnimatePresence,
    useScroll,
    useMotionValueEvent,
} from 'framer-motion';
import { cn } from '@/lib/utils';

export const FloatingNav = ({ className, onSelectionChange }) => {
    const { scrollYProgress } = useScroll();
    const [visible, setVisible] = useState(false);
    const [selectedOption, setSelectedOption] = useState('ask');

    useMotionValueEvent(scrollYProgress, 'change', (current) => {
        if (typeof current === 'number') {
            let direction = current - scrollYProgress.getPrevious();

            if (scrollYProgress.get() < 0.05) {
                setVisible(false);
            } else {
                if (direction < 0) {
                    setVisible(true);
                } else {
                    setVisible(false);
                }
            }
        }
    });

    const handleOptionClick = (option) => {
        setSelectedOption(option);
        onSelectionChange(option);
    };

    return (
        <AnimatePresence mode="wait">
            <motion.div
                initial={{
                    opacity: 1,
                    y: -100,
                }}
                animate={{
                    y: visible ? 0 : -100,
                    opacity: visible ? 1 : 0,
                }}
                transition={{
                    duration: 0.2,
                }}
                className={cn(
                    'flex max-w-fit fixed top-10 inset-x-0 mx-auto border border-transparent dark:border-white/[0.2] rounded-full dark:bg-black bg-white shadow-[0px_2px_3px_-1px_rgba(0,0,0,0.1),0px_1px_0px_0px_rgba(25,28,33,0.02),0px_0px_0px_1px_rgba(25,28,33,0.08)] z-[5000] pr-2 pl-8 py-2 items-center justify-center space-x-4',
                    className
                )}
            >
                {/* Ask Option */}
                <button
                    onClick={() => handleOptionClick('ask')}
                    className={cn(
                        'relative items-center flex space-x-1 text-sm px-4 py-2 rounded-full transition-colors duration-300',
                        selectedOption === 'ask'
                            ? 'bg-blue-500 text-white'
                            : 'text-neutral-600 dark:text-neutral-50 hover:bg-neutral-100'
                    )}
                >
                    Ask
                </button>

                {/* Overall Option */}
                <button
                    onClick={() => handleOptionClick('overall')}
                    className={cn(
                        'relative items-center flex space-x-1 text-sm px-4 py-2 rounded-full transition-colors duration-300',
                        selectedOption === 'overall'
                            ? 'bg-blue-500 text-white'
                            : 'text-neutral-600 dark:text-neutral-50 hover:bg-neutral-100'
                    )}
                >
                    Overall
                </button>
            </motion.div>
        </AnimatePresence>
    );
};
