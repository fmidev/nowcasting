var FC_PARAMS = {"Paine":"Pressure", "Geopotentiaalikorkeus":"GeopHeight", "Lämpötila":"Temperature", "Kastepiste":"DewPoint", "Kosteus":"Humidity", "Tuulen nopeus":"WindSpeedMS", "Tuulen suunta":"WindDirection", "Tuuli u-komponentti":"WindUMS", "Tuuli v-komponentti":"WindVMS" }
var STATS = {"RMSE":"rmse", "ME":"me"}
var VARYING_PARAMS = {"Ennustettavuus":"predictability", "Farneback winsize":"fbwinsize", "Farneback levels":"fblevel", "Farneback Poly ns":"fbpolyns"}


//20180913000000_20180913060000_me_adv_fbpolyns_WindVMS.png


$(function() {
    fillTables();
    selectParameters();
});

function fillTables() {
    var str1='';for (var i in FC_PARAMS){ str1 += '<option value="' + FC_PARAMS[i] + '">' + i + '</option>'};
    $("#sel1").html(str1);
    var str2='';for (var i in STATS){ str2 += '<option value="' + STATS[i] + '">' + i + '</option>'};
    $("#sel2").html(str2);
    var str3='';for (var i in VARYING_PARAMS){ str3 += '<option value="' + VARYING_PARAMS[i] + '">' + i + '</option>'};
    $("#sel3").html(str3);  
}

function selectParameters() {

    FC_PARAM = $("#sel1 option:selected").val();
    STAT = $("#sel2 option:selected").val();
    VARYING_PARAM = $("#sel3 option:selected").val();

    YEAR = STARTTIME.substr(0,4);
    MONTH = STARTTIME.substr(4,2);
    DAY = STARTTIME.substr(6,2);
    HOUR = STARTTIME.substr(8,2);
    MIN = STARTTIME.substr(10,2);
    SEC = STARTTIME.substr(12,2);

    var im = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_" + STAT +  "_adv_" + VARYING_PARAM + "_" + FC_PARAM + ".png";
    $("#fig").attr("src", im);

    var label = 'Forecast started at ' + YEAR + MONTH + DAY + ' ' + HOUR + ':' + MIN
    document.getElementById('label').innerHTML = label

}


