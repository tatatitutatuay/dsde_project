'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { parse } from 'csv-parse/browser/esm';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const ChoroplethMap = ({ keyword, color_num = 0 }) => {
    const [locations, setLocations] = useState([]);
    const [counts, setCounts] = useState([]);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true); // Set to true after the component mounts (client-side)
    }, []);

    const colorScales = {
        0: [
            [0, '#f4edfc'],
            [0.25, '#decaf7'],
            [0.5, '#c8a7f1'],
            [0.75, '#b283ec'],
            [1, '#9b5de5'],
        ], // Purples
        1: [
            [0, '#fdecf6'],
            [0.25, '#fcdaee'],
            [0.5, '#f8b4dd'],
            [0.75, '#f58ecc'],
            [1, '#f15bb5'],
        ], // Pinks
        2: [
            [0, '#fffceb'],
            [0.25, '#fef7c2'],
            [0.5, '#fef19a'],
            [0.75, '#feeb71'],
            [1, '#f5d000'],
        ], // Yellows
        3: [
            [0, '#ebfaff'],
            [0.25, '#c2f0ff'],
            [0.5, '#99e6ff'],
            [0.75, '#70dbff'],
            [1, '#00bbf9'],
        ], // Blues
        4: [
            [0, '#d6fff9'],
            [0.25, '#adfff3'],
            [0.5, '#85ffed'],
            [0.75, '#5cffe7'],
            [1, '#00e0bf'],
        ], // Greens
        5: [
            [0, '#e0e0e0'],
            [0.25, '#b8b8b8'],
            [0.5, '#8f8f8f'],
            [0.75, '#525252'],
            [1, '#292929'],
        ], // Blacks
    };

    useEffect(() => {
        // Fetch CSV file and parse using csv-parse
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

                        if (Array.isArray(keyword)) {
                            // If keyword is an array, sum up the counts for each keyword
                            keyword.forEach((key) => {
                                if (typeof key === 'string') {
                                    const keyLowerCase = key.toLowerCase();

                                    const dataForKey = result.filter(
                                        (row) =>
                                            row.keyword &&
                                            row.keyword
                                                .toLowerCase()
                                                .includes(keyLowerCase)
                                    );
                                    filteredData = [
                                        ...filteredData,
                                        ...dataForKey,
                                    ];
                                } else {
                                    console.error(
                                        `Invalid keyword type: ${typeof key}. Expected string.`
                                    );
                                }
                            });
                        } else if (keyword) {
                            // If keyword is a single string, process as before
                            const keywordLowerCase = keyword.toLowerCase();
                            filteredData = result.filter(
                                (row) =>
                                    row.keyword &&
                                    row.keyword
                                        .toLowerCase()
                                        .includes(keywordLowerCase)
                            );
                        } else {
                            console.error('Keyword is invalid or null');
                        }

                        // Aggregate the data for the filtered rows
                        const countryCounts = filteredData.reduce(
                            (acc, row) => {
                                const isoCode = row.iso_a3;
                                const count = parseInt(row.count, 10);
                                if (acc[isoCode]) {
                                    acc[isoCode] += count;
                                } else {
                                    acc[isoCode] = count;
                                }
                                return acc;
                            },
                            {}
                        );

                        // Map the aggregated data to locations and counts
                        const countryCodes = Object.keys(countryCounts);
                        const countryCountsValues =
                            Object.values(countryCounts);

                        setLocations(countryCodes);
                        setCounts(countryCountsValues);

                        console.log('locations:', countryCodes);
                        console.log('counts:', countryCountsValues);
                    }
                );
            })
            .catch((error) => {
                console.error('Error fetching CSV:', error);
            });
    }, [keyword]);

    const selectedColorscale = colorScales[color_num] || colorScales[0];

    const data = [
        {
            type: 'choroplethmap',
            locations: locations,
            z: counts,
            geojson:
                'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json',
            colorscale: selectedColorscale,
            colorbar: {
                title: 'Frequency',
                titleside: 'right',
            },
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
            t: 20,
            b: 20,
            l: 40,
            r: 40,
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

export default ChoroplethMap;
