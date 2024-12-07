'use client';

import { useState, useEffect } from 'react';

import { Button, Field, Textarea } from '@headlessui/react';
import { TypewriterEffect } from '@/components/ui/typewriter-effect';
import clsx from 'clsx';

import WordCloud from '@/components/WordCloud';
import BubbleMap from '@/components/BubbleMap';
import BarChart from '@/components/BarChart';
import NetworkComponent from '@/components/Network';

import { extractKeywords } from '@/lib/extractKeywords';

export default function Home() {
    const [abstract, setAbstract] = useState('');
    const [abstractDisabled, setAbstractDisabled] = useState(false);
    const [keywords, setKeywords] = useState([]);

    const handleA2K = async () => {
        try {
            if (!abstract) {
                alert('Please enter an abstract before submitting.');
                return;
            }

            console.log('Attempting to fetch keywords...');
            const response = await extractKeywords(abstract);

            const keywords = response;
            setKeywords(keywords);
            setAbstractDisabled(true);
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Failed to extract keywords. Please try again.');
        }
    };

    const handleClear = () => {
        setAbstract('');
        setKeywords([]);
        setAbstractDisabled(false);
    };

    const [isCopied, setIsCopied] = useState(false);
    const handleCopy = () => {
        const text = keywords.map(([word]) => word).join(', ');

        navigator.clipboard.writeText(text).then(() => {
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 3000);
        });
    };

    // Bubble Map
    const [populationData, setPopulationData] = useState(null);
    const [worldMap, setWorldMap] = useState(null);

    useEffect(() => {
        // Fetch population data and world map
        Promise.all([
            fetch('/data/world_population.json').then((res) => res.json()),
            fetch('/data/countries.geojson').then((res) => res.json()),
        ]).then(([populationData, worldMap]) => {
            setPopulationData(populationData);
            setWorldMap(worldMap);
        });
    }, []);

    // Network Visualization
    const [nodes, setNodes] = useState(null);
    const [edges, setEdges] = useState(null);

    const nodesTemp = [
        { id: 1, label: 'Node 1' },
        { id: 2, label: 'Node 2' },
        { id: 3, label: 'Node 3' },
    ];

    const edgesTemp = [
        { from: 1, to: 2 },
        { from: 1, to: 3 },
        { from: 2, to: 3 },
    ];

    useEffect(() => {
        setNodes(nodesTemp);
        setEdges(edgesTemp);
    }, []);

    return (
        <main className="container mx-auto my-24 flex justify-center items-center">
            <div className="w-full flex flex-col gap-12 items-center justify-center">
                {/* Title */}
                <div className="text-center flex flex-col gap-4">
                    <TypewriterEffect
                        words={[
                            { text: 'Drop' },
                            { text: 'Your' },
                            { text: 'Abstract', className: 'text-[#f15bb5]' },
                            { text: 'Here' },
                        ]}
                        cursorClassName="bg-black/80"
                    />
                    <p className="text-black/50 text-md text-pretty">
                        This tool will generate keywords for your paper using
                        data from Chulalongkorn University Library.
                    </p>
                </div>

                {/* Textarea */}
                <div className="flex flex-col justify-center items-end gap-4 max-w-2xl w-full">
                    <TextField
                        value={abstract}
                        onChange={(e) => {
                            setAbstract(e.target.value);
                        }}
                        placeholder={'Paste your abstract here...'}
                        disabled={abstractDisabled}
                    />
                    {abstractDisabled ? (
                        <Button2 onClick={handleClear}>Clear Abstract</Button2>
                    ) : (
                        <Button2 onClick={handleA2K}>Submit</Button2>
                    )}
                </div>

                {
                    // Keywords
                    keywords.length > 0 && (
                        <div className="flex flex-col gap-4 items-center">
                            <h3 className="text-2xl font-semibold text-black">
                                Keywords
                            </h3>
                            <div className="flex flex-wrap gap-4 justify-center items-center">
                                {keywords.map((keyword, index) => (
                                    <span
                                        key={index}
                                        className="bg-black/5 rounded-full px-3 py-1 text-sm/6 text-black"
                                    >
                                        {keyword[0]}
                                    </span>
                                ))}
                                <span key={'copy'}>
                                    <CopyButton
                                        handleCopy={handleCopy}
                                        isCopied={isCopied}
                                    />
                                </span>
                            </div>
                        </div>
                    )
                }

                {
                    // Bar Chart
                    keywords.length > 0 && (
                        <BarChart data={keywords.map((keyword) => keyword)} />
                    )
                }

                {
                    // Bubble Map
                    keywords.length > 0 && (
                        <BubbleMap
                            populationData={populationData}
                            worldMap={worldMap}
                        />
                    )
                }

                {/* Word Cloud */}
                <WordCloud
                    csvFilePath="/data/keyword_counts.csv"
                    minCount={20}
                />

                {/* Network Visualization*/}
                <NetworkComponent path="data\network_data.json"/>
            </div>
        </main>
    );
}

// TextField component
const TextField = ({ value, onChange, placeholder, disabled }) => {
    return (
        <div className="w-full max-w-2xl">
            <Field>
                <Textarea
                    className={clsx(
                        'mt-3 block w-full resize-none rounded-lg border-none bg-black/5 py-1.5 px-3 text-sm/6 text-black',
                        'focus:outline-none data-[focus]:outline-2 data-[focus]:-outline-offset-2 data-[focus]:outline-black/25'
                    )}
                    rows={5}
                    value={value}
                    onChange={onChange}
                    placeholder={placeholder}
                    disabled={disabled}
                />
            </Field>
        </div>
    );
};

// Button component
const Button2 = ({ children, onClick }) => {
    return (
        <Button
            className="inline-flex justify-center items-center gap-2 rounded-md bg-gray-900 py-1.5 px-3 text-sm/6 font-medium text-white max-w-36 text-center shadow-inner shadow-white/10 focus:outline-none data-[hover]:bg-gray-600 data-[open]:bg-gray-700 data-[focus]:outline-1 data-[focus]:outline-white"
            onClick={onClick}
        >
            {children}
        </Button>
    );
};

// Copy
const CopyButton = ({ handleCopy, isCopied }) => {
    return (
        <button
            onClick={handleCopy}
            className="text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg p-2 inline-flex items-center justify-center"
        >
            {!isCopied ? (
                <span id="default-icon">
                    <svg
                        className="w-3.5 h-3.5"
                        aria-hidden="true"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 18 20"
                    >
                        <path d="M16 1h-3.278A1.992 1.992 0 0 0 11 0H7a1.993 1.993 0 0 0-1.722 1H2a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2Zm-3 14H5a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2Zm0-4H5a1 1 0 0 1 0-2h8a1 1 0 1 1 0 2Zm0-5H5a1 1 0 0 1 0-2h2V2h4v2h2a1 1 0 1 1 0 2Z" />
                    </svg>
                </span>
            ) : (
                <span id="success-icon" className="inline-flex items-center">
                    <svg
                        className="w-3.5 h-3.5 text-blue-700 dark:text-blue-500"
                        aria-hidden="true"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 16 12"
                    >
                        <path
                            stroke="currentColor"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M1 5.917 5.724 10.5 15 1.5"
                        />
                    </svg>
                </span>
            )}
        </button>
    );
};
