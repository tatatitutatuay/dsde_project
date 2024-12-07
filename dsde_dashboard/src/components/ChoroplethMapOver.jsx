import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import Plot from 'react-plotly.js';

const ChoroplethMapOver = ({ keywords }) => {
    const [locations, setLocations] = useState([]);
    const [counts, setCounts] = useState([]);
    const [hoverData, setHoverData] = useState([]); // Store hover information

    if (!keywords) return <div>loading...</div>;

    useEffect(() => {
        // Parse the CSV file
        Papa.parse('/data/keyword_country.csv', {
            download: true,
            header: true,
            complete: (result) => {
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
                                const keywordLowerCase = keyword.toLowerCase();

                                // Filter data where the keyword is part of the "keyword" field
                                const dataForKey = result.data.filter(
                                    (row) =>
                                        row.keyword &&
                                        row.keyword
                                            .toLowerCase()
                                            .includes(keywordLowerCase)
                                );
                                filteredData = [...filteredData, ...dataForKey];
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
                const countryCounts = filteredData.reduce((acc, row) => {
                    const isoCode = row.iso_a3;
                    const keyword = row.keyword;
                    const count = parseInt(row.count, 10);

                    // Find the matching keyword in the keywords parameter (e.g., 'science')
                    keywords.forEach(([keywordMatch]) => {
                        if (
                            keyword
                                .toLowerCase()
                                .includes(keywordMatch.toLowerCase())
                        ) {
                            const adjustedCount = count;

                            if (!acc[isoCode]) acc[isoCode] = {};
                            if (!acc[isoCode][keywordMatch])
                                acc[isoCode][keywordMatch] = 0;

                            acc[isoCode][keywordMatch] += adjustedCount;
                        }
                    });

                    return acc;
                }, {});

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
                        .map(([keyword, count]) => `${keyword}: ${count}`)
                        .join('<br>'); // Join keyword counts by line breaks
                });

                setLocations(countryCodes);
                setCounts(countryCountsValues);
                setHoverData(hoverInfo);

                console.log('locations:', countryCodes);
                console.log('counts:', countryCountsValues);
                console.log('hoverData:', hoverInfo);
            },
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
            <Plot data={data} layout={layout} />
        </div>
    );
};

export default ChoroplethMapOver;
