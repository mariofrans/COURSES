from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

##################################################################################################################

""" WRITING EFFICIENT QUERIES """

"""
We will use two functions to compare the efficiency of different queries:
    1. show_amount_of_data_scanned() shows the amount of data the query uses.
    2. show_time_to_run() prints how long it takes for the query to execute.
"""

##################################################################################################################

""" STEP 1: YOU WORK FOR PET COSTUMES INTERNATIONAL """

"""
You need to write three queries this afternoon. You have enough time to write working versions of all three, but 
only enough time to think about optimizing one of them. Which of these queries is most worth optimizing?

    1. A software engineer wrote an app for the shipping department, to see what items need to be shipped and which 
    aisle of the warehouse to go to for those items. She wants you to write the query. It will involve data that 
    is stored in an orders table, a shipments table and a warehouseLocation table. The employees in the shipping 
    department will pull up this app on a tablet, hit refresh, and your query results will be shown in a nice 
    interface so they can see what costumes to send where.

    2. The CEO wants a list of all customer reviews and complaints… which are conveniently stored in a single reviews 
    table. Some of the reviews are really long… because people love your pirate costumes for parrots, and they can’t 
    stop writing about how cute they are.

    3. Dog owners are getting more protective than ever. So your engineering department has made costumes with 
    embedded GPS trackers and wireless communication devices. They send the costumes’ coordinates to your database 
    once a second. You then have a website where owners can find the location of their dogs (or at least the costumes 
    they have for those dogs). For this service to work, you need a query that shows the most recent location for all 
    costumes owned by a given human. This will involve data in a CostumeLocations table as well as a CostumeOwners 
    table.

So, which of these could benefit most from being written efficiently? Set the value of the query_to_optimize variable 
below to one of 1, 2, or 3. (Your answer should have type integer.)
"""

query_to_optimize = 3

"""
Why 3: Because data is sent for each costume at each second, this is the query that is likely to involve the most 
data (by far). And it will be run on a recurring basis. So writing this well could pay off on a recurring basis.

Why not 1: This is the second most valuable query to optimize. It will be run on a recurring basis, and it involves 
merges, which is commonly a place where you can make your queries more efficient

Why not 2: This sounds like it will be run only one time. So, it probably doesn’t matter if it takes a few seconds 
extra or costs a few cents more to run that one time. Also, it doesn’t involve JOINs. While the data has text 
fields (the reviews), that is the data you need. So, you can’t leave these out of your select query to save 
computation.
"""

##################################################################################################################

""" STEP 2: MAKE IT EASIER TO FIND MITZIE! """

"""
You have the following two tables:

The CostumeLocations table shows timestamped GPS data for all of the pet costumes in the database, where 
CostumeID is a unique identifier for each costume.

The CostumeOwners table shows who owns each costume, where the OwnerID column contains unique identifiers for 
each (human) owner. Note that each owner can have more than one costume! And, each costume can have more than 
one owner: this allows multiple individuals from the same household (all with their own, unique OwnerID) to 
access the locations of their pets' costumes.

Say you need to use these tables to get the current location of one pet in particular: Mitzie the Dog recently 
ran off chasing a squirrel, but thankfully she was last seen in her hot dog costume!

One of Mitzie's owners (with owner ID MitzieOwnerID) logs into your website to pull the last locations of every 
costume in his possession. Currently, you get this information by running the following query:

Is there a way to make this faster or cheaper?
"""

query = """
        WITH LocationsAndOwners AS 
        (
        SELECT * 
        FROM CostumeOwners co INNER JOIN CostumeLocations cl
        ON co.CostumeID = cl.CostumeID
        ),
        LastSeen AS
        (
        SELECT CostumeID, MAX(Timestamp)
        FROM LocationsAndOwners
        GROUP BY CostumeID
        )
        SELECT lo.CostumeID, Location 
        FROM LocationsAndOwners lo INNER JOIN LastSeen ls 
            ON lo.Timestamp = ls.Timestamp AND lo.CostumeID = ls.CostumeID
        WHERE OwnerID = MitzieOwnerID
        """

"""
Yes. Working with the LocationsAndOwners table is very inefficient, because it’s a big table. There are a few 
options here, and which works best depends on database specifics. One likely improvement is:
"""


query = """
        WITH CurrentOwnersCostumes AS
        (
        SELECT CostumeID 
        FROM CostumeOwners 
        WHERE OwnerID = MitzieOwnerID
        ),
        OwnersCostumesLocations AS
        (
        SELECT cc.CostumeID, Timestamp, Location 
        FROM CurrentOwnersCostumes cc INNER JOIN CostumeLocations cl
            ON cc.CostumeID = cl.CostumeID
        ),
        LastSeen AS
        (
        SELECT CostumeID, MAX(Timestamp)
        FROM OwnersCostumesLocations
        GROUP BY CostumeID
        )
        SELECT ocl.CostumeID, Location 
        FROM OwnersCostumesLocations ocl INNER JOIN LastSeen ls 
            ON ocl.timestamp = ls.timestamp AND ocl.CostumeID = ls.costumeID
        """

"""
Why is this better?
Instead of doing large merges and running calculations (like finding the last timestamp) for every costume, we 
discard the rows for other owners as the first step. So each subsequent step (like calculating the last timestamp) 
is working with something like 99.999% fewer rows than what was needed in the original query.
Databases have something called “Query Planners” to optimize details of how a query executes even after you write 
it. Perhaps some query planner would figure out the ability to do this. But the original query as written would 
be very inefficient on large datasets.
"""

##################################################################################################################