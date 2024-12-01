// import * as d3 from 'd3';
// import * as topojson from 'topojson-client'; // If you're using TopoJSON format

// const WorldMap = ({ width, height, geoData, populationData }) => {
//     const centroid = (feature) => {
//         const path = d3.geoPath();
//         return path.centroid(feature); // This calculates the centroid of the feature
//     }

//     // Join the geographic shapes (geoData) and the population data.
//     const data = populationData.map((d) => ({
//         ...d,
//         countyData: geoData.features.find(f => f.properties.fips === d.fips) // Match by fips
//     }))
//     .filter(d => d.countyData) // Only include those with matching geographical data
//     .sort((a, b) => d3.descending(a.population, b.population));

//     // Construct the radius scale for population size
//     const radius = d3.scaleSqrt([0, d3.max(data, d => d.population)], [0, 40]);

//     // Construct a path generator
//     const path = d3.geoPath();

//     // Create the SVG container with width and height
//     const svg = d3.create('svg')
//         .attr('width', width)
//         .attr('height', height)
//         .attr('viewBox', [0, 0, width, height])
//         .attr('style', 'width: 100%; height: auto; height: intrinsic;');

//     // Create the cartographic background layers
//     svg.append('path')
//         .datum(topojson.feature(geoData, geoData.objects.nation)) // Assuming geoData is in TopoJSON format
//         .attr('fill', '#ddd')
//         .attr('d', path);

//     svg.append('path')
//         .datum(topojson.mesh(geoData, geoData.objects.states, (a, b) => a !== b))
//         .attr('fill', 'none')
//         .attr('stroke', 'white')
//         .attr('stroke-linejoin', 'round')
//         .attr('d', path);

//     // Create the legend for circle sizes (population)
//     const legend = svg.append('g')
//         .attr('fill', '#777')
//         .attr('transform', 'translate(915,608)')
//         .attr('text-anchor', 'middle')
//         .style('font', '10px sans-serif')
//         .selectAll()
//         .data(radius.ticks(4).slice(1))
//         .join('g');

//     legend.append('circle')
//         .attr('fill', 'none')
//         .attr('stroke', '#ccc')
//         .attr('cy', d => -radius(d))
//         .attr('r', radius);

//     legend.append('text')
//         .attr('y', d => -2 * radius(d))
//         .attr('dy', '1.3em')
//         .text(radius.tickFormat(4, 's'));

//     // Add a circle for each population entry, with a tooltip showing the county and population
//     const format = d3.format(',.0f');
//     svg.append('g')
//         .attr('fill', 'brown')
//         .attr('fill-opacity', 0.5)
//         .attr('stroke', '#fff')
//         .attr('stroke-width', 0.5)
//         .selectAll()
//         .data(data)
//         .join('circle')
//         .attr('transform', d => `translate(${centroid(d.countyData)})`)
//         .attr('r', d => radius(d.population))
//         .append('title')
//         .text(d => `${d.countyData.properties.name}, ${d.countyData.properties.state}\nPopulation: ${format(d.population)}`);

//     return svg.node();
// };

// export default WorldMap;
