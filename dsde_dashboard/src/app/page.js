'use client';

import LinePlot from '@/components/LinePlot';
import WordCloud from '@/components/WordCloud';
import WorldMap from '@/components/WorldMap';
import { numData } from '@/data/data';
import { geoData } from '@/data/geoData';

const data = [
    { country: 'USA', date: '2023-05-10' },
    { country: 'Canada', date: '2022-07-15' },
    { country: 'Mexico', date: '2023-01-20' },
];

const data2 = [
    ['React', 10],
    ['JavaScript', 8],
    ['CSS', 6],
    ['HTML', 4],
];

const wordArray = Array.from({ length: 200 }, () => {
    const words = [
        'apple',
        'banana',
        'cherry',
        'date',
        'elderberry',
        'fig',
        'grape',
        'honeydew',
        'kiwi',
        'lemon',
        'mango',
        'nectarine',
        'orange',
        'papaya',
        'quince',
        'raspberry',
        'strawberry',
        'tomato',
        'watermelon',
    ];
    const word = words[Math.floor(Math.random() * words.length)]; // Randomly pick a word
    const count = Math.floor(Math.random() * 10) + 1; // Random count between 1 and 10
    return [word, count];
});

export default function Home() {
    // Call WordCloud and get the SVG node
    const wordCloudSVG = WordCloud(wordArray, {
        width: 2500,
        height: 1000,
        size: () => 0.3 + Math.random(),
    });

    return (
        <main className="flex flex-col gap-8 items-center sm:items-start font-[family-name:var(--font-geist-sans)]">
            <LinePlot data={[10, 20, 15, 30, 25]} />
            {/* <WorldMap
                geoData={geoData}
                numData={numData}
                width={1000}
                height={1000}
            /> */}
            {/* Render WordCloud SVG directly */}
            <div
                className="word-cloud-container"
                dangerouslySetInnerHTML={{ __html: wordCloudSVG.outerHTML }}
            />
        </main>
    );
}
