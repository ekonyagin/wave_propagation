import numpy as np

params_dict_cartesian = {
						    "mode_0": 	[5., 1, 0, 1., 3.**2*np.pi],
						    "mode_1": 	[5., 2, 0, 1., 3.**2*np.pi],
						    "coords":
						        {
						            "type": 	"cartesian",
						            "xmin": 	-10,
						            "xmax": 	10,
						            "ymin": 	-10,
						            "ymax": 	10,
						            "zmin": 	-15,
						            "zmax": 	15
						        },
						    "tau": 	4*2*np.pi,
						    "delay": 	-2*4*2*np.pi,
						    "discretization": 	{
						                        	"x": 	60,
						                        	"y": 	60,
						                        	"z": 	2400
						    					}
						}

params_dict_polar = 	{
	    					"mode_0" : [5., 1, 0, 1., 3.**2*np.pi],
	    					"mode_1" : [5., 2, 0, 1., 3.**2*np.pi],
	    					"coords" :
	        							{
	            							"type": "polar",
	            							"zmin": -15,
	            							"zmax": 15
	        							},
	    					"tau": 		0.5*2*np.pi,
	    					"delay": 	-2*2*np.pi,
	    					"discretization": 	{
	                        						"z": 	4800,
	                        						"rho":	20,
	                        						"phi": 150
	    										}
               			}