import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const BarChart = ({ data }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!data) return <div> No Data </div>;

        const margin = { top: 20, right: 30, bottom: 40, left: 40 };
        const width = 500 - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        // Create the SVG container
        const svg = d3
            .select(svgRef.current)
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Create scales for the X and Y axes
        const x = d3
            .scaleBand()
            .domain(data.map((d) => d[0])) // Use the keyword as the domain
            .range([0, width])
            .padding(0.1);

        const y = d3
            .scaleLinear()
            .domain([0, d3.max(data, (d) => d[1])]) // Use the maximum value for the Y domain
            .nice() // Add some nice rounding to the axis
            .range([height, 0]);

        // Color array for each bar
        const colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', '#00f5d4'];

        // Create the bars
        svg.selectAll('.bar')
            .data(data)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', (d, i) => x(d[0])) // Position the bar along the x-axis
            .attr('y', (d) => y(d[1])) // Position the bar along the y-axis
            .attr('width', x.bandwidth()) // Set the width of the bar
            .attr('height', (d) => height - y(d[1])) // Set the height of the bar
            .attr('fill', (d, i) => colors[i]); // Set the fill color based on the index

        // Create the X-axis
        svg.append('g')
            .selectAll('.x-axis')
            .data([0])
            .enter()
            .append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x));

        // Create the Y-axis
        svg.append('g').call(d3.axisLeft(y));
    }, [data]);

    return <svg ref={svgRef}></svg>;
};

export default BarChart;
