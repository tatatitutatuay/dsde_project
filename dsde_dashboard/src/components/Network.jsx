'use client'

import React, { useEffect, useRef } from 'react';
import { Network } from 'vis-network';

export default function NetworkComponent({
    nodesData, 
    edgesData
}) {
    const networkContainerRef = useRef(null);

    useEffect(() => {
        // Options for the network
        const options = {
            nodes: {
                shape: "dot",
                size: 16,
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
        new Network(networkContainerRef.current, { nodes: nodesData, edges: edgesData }, options);
    }, [nodesData, edgesData]);

    return (
        <div
            ref={networkContainerRef}
            style={{ height: "750px", width: "100%", border: "1px solid black" }}
        />
    );
};
