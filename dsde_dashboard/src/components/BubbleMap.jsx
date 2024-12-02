import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

// BubbleMap component for World Map
const BubbleMap = ({ populationData, worldMap }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!populationData || !worldMap) {
            console.log('Invalid data');
            return;
        }

        const width = 975;
        const height = 610;

        // Map projection
        const projection = d3.geoMercator().scale(150).translate([width / 2, height / 2]);
        const path = d3.geoPath().projection(projection);

        // Map data processing (exclude Antarctica)
        const data = populationData
            .map((d) => ({
                ...d,
                country: worldMap.features.find(
                    (feature) => feature.properties.ISO_A3 === d.iso_a3 && d.iso_a3 !== 'ATA' // Exclude Antarctica (ISO_A3: ATA)
                ),
            }))
            .filter((d) => d.country) // Ensure that only valid countries are included

        // Define color and opacity for each keyword
        const keywordStyle = (idx) => {
            switch (idx % 3) {
                case 0:
                    return { fill: '#9b5de5', fillOpacity: 0.2, stroke: '#9b5de5', strokeOpacity: 1 };
                case 1:
                    return { fill: '#fee440', fillOpacity: 0.2, stroke: '#fee440', strokeOpacity: 1 };
                case 2:
                    return { fill: '#00bbf9', fillOpacity: 0.2, stroke: '#00bbf9', strokeOpacity: 1 };
                default:
                    return { fill: 'gray', fillOpacity: 0.2, stroke: 'gray', strokeOpacity: 1 };
            }
        };

        // Clear existing SVG if re-rendered
        d3.select(svgRef.current).selectAll('*').remove();

        const svg = d3
            .select(svgRef.current)
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height])
            .attr('style', 'width: 100%; height: auto; height: intrinsic;');

        // Draw land background using GeoJSON directly
        svg.append('path')
            .datum(worldMap)
            .attr('fill', '#ddd')  // Land color
            .attr('d', path);

        // Draw country borders
        svg.append('g')
            .selectAll('path')
            .data(worldMap.features.filter(feature => feature.properties.ISO_A3 !== 'ATA')) // Exclude Antarctica
            .join('path')
            .attr('fill', 'none')
            .attr('stroke', 'white')
            .attr('stroke-linejoin', 'round')
            .attr('d', path);

        // Draw circles for each keyword in each country with offset for separate circles
        const format = d3.format(',.0f');

        svg.append('g')
            .selectAll('circle')
            .data(data.flatMap((d) =>
                d.keyword.map((k, idx) => ({
                    country: d.country,
                    keyword: k,
                    iso_a3: d.iso_a3,
                    offset: idx, // Use the index to add different offset for each circle
                }))
            ))
            .join('circle')
            .attr('transform', (d) => {
                const centroid = path.centroid(d.country);
                // Add a small offset to each circle based on keyword index
                const offset = 10 * d.offset; // Adjust the multiplier as needed
                return `translate(${centroid[0]}, ${centroid[1]})`;
            })
            .attr('r', (d) => {
                // Adjust radius based on the frequency of the keyword
                const radiusScale = d3.scaleSqrt().domain([0, d3.max(data.flatMap((d) => d.keyword.map((k) => k.frequency)))]).range([0, 40]);
                return radiusScale(d.keyword.frequency);
            })
            .attr('fill', (d, i) => keywordStyle(i).fill) // Set fill color
            .attr('fill-opacity', (d, i) => keywordStyle(i).fillOpacity) // Set fill opacity
            .attr('stroke', (d, i) => keywordStyle(i).stroke) // Set stroke color
            .attr('stroke-opacity', (d, i) => keywordStyle(i).strokeOpacity) // Set stroke opacity
            .append('title')
            .text(
                (d) =>
                    `${d.keyword.name} in ${d.country.properties.ADMIN}\nFrequency: ${format(d.keyword.frequency)}`
            );

        // Draw legend in the top-right corner
        const legend = svg
            .append('g')
            .attr('fill', '#777')
            .attr('transform', `translate(${width - 100}, 40)`) // Move legend to top-right
            .attr('text-anchor', 'middle')
            .style('font', '10px sans-serif')
            .selectAll('g')
            .data([0, 1, 2]) // Data for the three keywords
            .join('g');

        legend
            .append('circle')
            .attr('fill', (d) => keywordStyle(d).fill)
            .attr('cy', (d, i) => i * 30) // Spacing between circles
            .attr('r', 6);

        legend
            .append('text')
            .attr('y', (d, i) => i * 30)
            .attr('dy', '1.3em')
            .text((d) => `Keyword ${d + 1}`);
    }, [populationData, worldMap]);

    return <svg ref={svgRef}></svg>;
};

export default BubbleMap;
