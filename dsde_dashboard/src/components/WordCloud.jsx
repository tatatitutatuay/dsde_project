'use client';

import * as d3 from 'd3';
import d3Cloud from 'd3-cloud';
import { useEffect, useRef, useState } from 'react';

export default function WordCloud({
    csvFilePath = '/path/to/keyword_counts.csv', // Path to the CSV file
    size = (d) => d.size,
    width = 1280,
    height = 500,
    fontFamily = 'sans-serif',
    fontScale = 15,
    minFontSize = 10,
    maxFontSize = 100,
    padding = 5,
    rotate = 0,
    maxWords = 250,
    minCount = 20, // Minimum count threshold for displaying words
}) {
    const [wordsArray, setWordsArray] = useState([]);
    const svgRef = useRef(null);

    useEffect(() => {
        // Load the CSV file and parse it
        const loadCSV = async () => {
            const data = await d3.csv(csvFilePath);
            const formattedData = data
                .map((d) => [d.Keyword, +d.Count])
                .filter(([, count]) => count >= minCount); // Filter out keywords with count less than minCount
            setWordsArray(formattedData);
        };

        loadCSV();
    }, [csvFilePath, minCount]);

    // Function to determine text color based on count
    const getTextColor = (count) => {
        if (count < 40) return 'gray';
        if (count >= 40 && count < 70) return '#f7cad0';
        if (count >= 70 && count < 100) return '#ff7096';
        if (count >= 100) return '#ff0a54';
        return 'black'; // Default color if outside the specified range
    };

    useEffect(() => {
        if (!svgRef.current || wordsArray.length === 0) return;

        const data = wordsArray
            .sort(([, a], [, b]) => d3.descending(a, b))
            .slice(0, maxWords)
            .map(([text, count]) => ({ text, count }));

        // Adjust font size scaling based on the count range
        const sizeScale = d3
            .scaleLinear()
            .domain([
                d3.min(data, (d) => d.count),
                d3.max(data, (d) => d.count),
            ]) // Ensure range covers min to max size
            .range([minFontSize, maxFontSize]);

        const svg = d3
            .select(svgRef.current)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('width', width)
            .attr('height', height)
            .attr('font-family', fontFamily)
            .attr('text-anchor', 'middle');

        const g = svg
            .append('g')
            .attr('transform', `translate(${width / 2},${height / 2})`);

        const cloud = d3Cloud()
            .size([width, height])
            .words(data)
            .padding(padding)
            .rotate(rotate)
            .font(fontFamily)
            .fontSize((d) => sizeScale(d.count)) // Scale the font size based on the keyword count
            .on('end', (words) => {
                g.selectAll('text')
                    .data(words)
                    .enter()
                    .append('text')
                    .attr('font-size', (d) => `${d.size}px`)
                    .attr('fill', (d) => getTextColor(d.count)) // Use the correct count for text color
                    .attr(
                        'transform',
                        (d) => `translate(${d.x},${d.y}) rotate(${d.rotate})`
                    )
                    .text((d) => d.text);
            });

        cloud.start();

        // Cleanup SVG on component unmount
        return () => {
            svg.selectAll('*').remove();
        };
    }, [
        wordsArray,
        width,
        height,
        fontFamily,
        fontScale,
        minFontSize,
        maxFontSize,
        padding,
        rotate,
        maxWords,
    ]);

    return <svg ref={svgRef}></svg>;
}
