'use client';

import * as d3 from 'd3';
import d3Cloud from 'd3-cloud';
import { useEffect, useRef } from 'react';

export default function WordCloud({
    wordsArray,
    size = (d) => d.size,
    width = 640,
    height = 400,
    fontFamily = 'sans-serif',
    fontScale = 15,
    minFontSize = 10,
    maxFontSize = 100,
    fill = d3.scaleOrdinal(d3.schemeCategory10),
    padding = 5,
    rotate = 0,
    maxWords = 250,
}) {
    const svgRef = useRef(null);

    useEffect(() => {
        if (!svgRef.current) return;

        const data = wordsArray
            .sort(([, a], [, b]) => d3.descending(a, b))
            .slice(0, maxWords)
            .map(([text, size]) => ({ text, size }));

        const sizeScale = d3
            .scaleLinear()
            .domain([0, d3.max(data, (d) => d.size)])
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
            .fontSize((d) => sizeScale(d.size))
            .on('end', (words) => {
                g.selectAll('text')
                    .data(words)
                    .enter()
                    .append('text')
                    .attr('font-size', (d) => `${d.size}px`)
                    .attr('fill', (d) => fill(d.text))
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
        fill,
        padding,
        rotate,
        maxWords,
    ]);

    return <svg ref={svgRef}></svg>;
}
