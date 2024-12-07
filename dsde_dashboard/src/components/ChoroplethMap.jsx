import React from 'react';
import Plot from 'react-plotly.js';

const BarChart = ({ data }) => {
    if (!data) return null;

    // Extract the categories and values from the data
    const categories = data.map((d) => d[0]);
    const values = data.map((d) => d[1]);

    return (
        <Plot
            data={[
                {
                    type: 'bar',
                    x: categories,
                    y: values,
                    marker: {
                        color: [
                            '#9b5de5',
                            '#f15bb5',
                            '#fee440',
                            '#00bbf9',
                            '#00f5d4',
                        ],
                    },
                },
            ]}
            layout={{
                title: 'Bar Chart',
                xaxis: {
                    title: 'Category',
                },
                yaxis: {
                    title: 'Value',
                },
                margin: {
                    t: 20,
                    r: 30,
                    b: 40,
                    l: 40,
                },
            }}
        />
    );
};

export default BarChart;