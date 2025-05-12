function printVar(file, varName, varData)

	%    file   	 file path where to save data
	%    varName  	 name of the variable to save
	%    varData     value of the variable to save
	
	% by G. Gusso - gabriele.gusso@roma1.infn.it
	% Department of Physics - Sapienza University - Rome

	if varName == "h0"
    	fprintf(file, '%s,%e\n', varName, varData);
	elseif mod(varData,1)==0
		fprintf(file, '%s,%i\n', varName, varData);
	else
		fprintf(file, '%s,%f\n', varName, varData);
	end
end
