'use client'

import React, { useEffect, useState, useRef } from 'react';
import { Network } from 'vis-network';

export default function NetworkComponent({
    path,
}) {
    const networkContainerRef = useRef(null);
    const [nodes, setNodes] = useState(null);
    const [edges, setEdges] = useState(null);

    // Fetch the data from the provided path
    useEffect(() => {
        fetch(path) // Update with your actual JSON path
            .then((response) => response.json())
            .then((data) => { 
                console.log(data); // Debugging: check the fetched data
                setNodes(data.nodes);
                setEdges(data.edges);
            });
    }, [path]);

    // Initialize the network after nodes and edges are loaded
    useEffect(() => {
        if (nodes && edges) { // Ensure nodes and edges are not null
            const options = {
                nodes: {
                    shape: "dot",
                    font: {
                        size: 16,
                    },
                },
                edges: {
                    color: "#848484",
                },
                physics: {
                    enabled: true,
                },
            };

            // Create the network visualization
            new Network(networkContainerRef.current, { nodes, edges }, options);
        }
    }, [nodes, edges]); // Re-run when nodes or edges change
    return (
        <div
            ref={networkContainerRef}
            style={{ height: "750px", width: "100%", border: "1px solid black" }}
        />
    );
};
