'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { parse } from 'csv-parse/browser/esm'; // Import csv-parse for browser

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const ChoroplethMapOver = ({ keywords }) => {
    const [locations, setLocations] = useState([]);
    const [counts, setCounts] = useState([]);
    const [hoverData, setHoverData] = useState([]);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true); // Set to true after the component mounts (client-side)
    }, []);

    if (!keywords) return <div>loading...</div>;

    useEffect(() => {
        // Fetch and parse the CSV file using csv-parse
        fetch('/data/keyword_country.csv')
            .then((response) => response.text())
            .then((csvData) => {
                parse(
                    csvData,
                    {
                        columns: true,
                        skip_empty_lines: true,
                    },
                    (err, result) => {
                        if (err) {
                            console.error('Error parsing CSV:', err);
                            return;
                        }

                        let filteredData = [];

                        if (Array.isArray(keywords)) {
                            // If keywords is an array of tuples, use the first element (keyword) for filtering
                            keywords.forEach((keywordTuple) => {
                                if (
                                    Array.isArray(keywordTuple) &&
                                    keywordTuple.length === 2
                                ) {
                                    const keyword = keywordTuple[0]; // Get the first element (keyword)
                                    if (typeof keyword === 'string') {
                                        const keywordLowerCase =
                                            keyword.toLowerCase();

                                        // Filter data where the keyword is part of the "keyword" field
                                        const dataForKey = result.filter(
                                            (row) =>
                                                row.keyword &&
                                                row.keyword
                                                    .toLowerCase()
                                                    .includes(keywordLowerCase)
                                        );
                                        filteredData = [
                                            ...filteredData,
                                            ...dataForKey,
                                        ];
                                    } else {
                                        console.error(
                                            `Invalid keyword type: ${typeof keyword}. Expected string.`
                                        );
                                    }
                                } else {
                                    console.error(
                                        `Invalid keyword tuple: ${JSON.stringify(keywordTuple)}`
                                    );
                                }
                            });
                        } else {
                            console.error('Keywords is invalid or null');
                        }

                        // Aggregate the data for the filtered rows
                        const countryCounts = filteredData.reduce(
                            (acc, row) => {
                                const isoCode = row.iso_a3;
                                const keyword = row.keyword;
                                const count = parseInt(row.count, 10);

                                // Find the matching keyword in the keywords parameter (e.g., 'science')
                                keywords.forEach(([keywordMatch]) => {
                                    if (
                                        keyword
                                            .toLowerCase()
                                            .includes(
                                                keywordMatch.toLowerCase()
                                            )
                                    ) {
                                        const adjustedCount = count;

                                        if (!acc[isoCode]) acc[isoCode] = {};
                                        if (!acc[isoCode][keywordMatch])
                                            acc[isoCode][keywordMatch] = 0;

                                        acc[isoCode][keywordMatch] +=
                                            adjustedCount;
                                    }
                                });

                                return acc;
                            },
                            {}
                        );

                        // Map the aggregated data to locations, counts, and hover info
                        const countryCodes = Object.keys(countryCounts);
                        const countryCountsValues = countryCodes.map((code) =>
                            Object.values(countryCounts[code]).reduce(
                                (total, count) => total + count,
                                0
                            )
                        );

                        // Prepare hover info with aggregated keyword counts for each country
                        const hoverInfo = countryCodes.map((code) => {
                            return Object.entries(countryCounts[code])
                                .map(
                                    ([keyword, count]) => `${keyword}: ${count}`
                                )
                                .join('<br>'); // Join keyword counts by line breaks
                        });

                        setLocations(countryCodes);
                        setCounts(countryCountsValues);
                        setHoverData(hoverInfo);

                        console.log('locations:', countryCodes);
                        console.log('counts:', countryCountsValues);
                        console.log('hoverData:', hoverInfo);
                    }
                );
            })
            .catch((error) => {
                console.error('Error fetching CSV:', error);
            });
    }, [keywords]);

    const data = [
        {
            type: 'choroplethmap',
            locations: locations, // ISO country codes
            z: counts, // Data values
            geojson:
                'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json',
            colorscale: [
                [0, '#e0e0e0'],
                [0.25, '#b8b8b8'],
                [0.5, '#8f8f8f'],
                [0.75, '#525252'],
                [1, '#292929'],
            ],
            colorbar: {
                title: 'Frequency',
                titleside: 'right',
            },
            text: hoverData, // Include hover data for each country
            hoverinfo: 'location+z+text', // Display location, frequency, and hover data
        },
    ];

    const layout = {
        geo: {
            showcoastlines: true,
            coastlinecolor: 'rgb(255, 255, 255)',
            projection: {
                type: 'mercator',
            },
        },
        width: 1000,
        height: 600,
        margin: {
            t: 20, // top margin
            b: 20, // bottom margin
            l: 40, // left margin
            r: 40, // right margin
        },
    };

    return (
        <div>
            {isClient && (
                <Plot
                    data={data}
                    layout={layout}
                    style={{ width: '100%', height: '100%' }}
                />
            )}
        </div>
    );
};

export default ChoroplethMapOver;
